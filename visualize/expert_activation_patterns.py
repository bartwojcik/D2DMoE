import logging
from collections import defaultdict
from pathlib import Path

import seaborn as sns
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from architectures.moe.moefication import MoeficationMoE
from datasets_config import DATASETS_NAME_MAP
from utils import configure_logging, get_loader, load_model


def get_default_args():
    default_args = OmegaConf.create()
    default_args.runs_dir = Path.cwd() / "runs"  # Root dir where experiment data was saved.
    default_args.exp_names = []  # Unique experiment names to visualize the results for (excluding exp_id).
    default_args.exp_ids = [0]  # Experiment ids.
    default_args.display_names = None  # Pretty display names that will be used when generating the plot.
    default_args.output_dir = Path.cwd() / "figures"  # Target directory.
    default_args.batch_size = None  # Batch size.
    default_args.dsti_expert_selection_mode = None  # Expert selection method for evaluation step
    default_args.dsti_tau_to_eval = None  # Tau threshold for selecting experts to execute
    default_args.dataset = None  # Dataset to evaluate on. If "None", will use the dataset used by the run.
    default_args.dataset_args = None  # Dataset arguments. If "None", will use same args as in the training run.
    default_args.use_wandb = False  # Use W&B. Will save and load the models from the W&B cloud.
    default_args.mixed_precision = None
    return default_args


def process_dataset(model, data_loader):
    moe_executed_experts = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for x, y in data_loader:
            _, gating_data = model(x, return_gating_data=True)
            # each element of gating_data_list is a tuple
            # because different MoEs classes can return more than simply the gating network's final outputs
            # we select only the final routing decisions
            gating_data = {k: v[0] for k, v in gating_data.items()}
            # gating data should be a dict with tensor values of size (batch_size, sequence_length, num_experts) now
            for moe_name, g in gating_data.items():
                moe_executed_experts[moe_name].append(((g > 0.0).int().sum(dim=-1) / g.size(-1)).detach().cpu())
    for moe_name in moe_executed_experts.keys():
        moe_executed_experts[moe_name] = torch.cat(moe_executed_experts[moe_name], dim=0)
    return moe_executed_experts


def set_for_eval_with_dynk(model, tau, mode=None):
    model.eval()
    for m in model.modules():
        if isinstance(m, MoeficationMoE):
            m.forward_mode = "dynk" if mode is None else mode
            m.tau = tau
    return model


def setup_and_process(accelerator, args, model, run_args):
    dataset = args.dataset if args.dataset is not None else run_args.dataset
    dataset_args = args.dataset_args if args.dataset_args is not None else run_args.dataset_args
    _, _, eval_data = DATASETS_NAME_MAP[dataset](**dataset_args)
    batch_size = args.batch_size if args.batch_size is not None else run_args.batch_size
    dataloader = get_loader(eval_data, batch_size, accelerator)
    model = accelerator.prepare(model)
    dsti_expert_selection_mode = args.dsti_expert_selection_mode if args.dsti_expert_selection_mode is not None else run_args.dsti_expert_selection_mode
    model = set_for_eval_with_dynk(model, args.dsti_tau_to_eval, dsti_expert_selection_mode)
    moe_executed_experts = process_dataset(model, dataloader)
    return moe_executed_experts


def plot_ratio_of_selected_experts_heatmap(moe_executed_experts, substructure=None, bins=100):
    saved_args = locals()
    histograms = []
    for moe_name in moe_executed_experts.keys():
        if substructure is not None and substructure not in moe_name:
            continue
        layer_histogram, bin_edges = torch.histogram(moe_executed_experts[moe_name].flatten(), bins=bins, range=(0, 1), density=True)
        histograms.append(layer_histogram)
    histograms = torch.stack(histograms, dim=0).log1p()
    num_layers = len(histograms)
    fig, ax = plt.subplots(figsize=(16, 12), facecolor="w", edgecolor="k")
    sns.heatmap(histograms, square=False, yticklabels=2, xticklabels=20, cbar=False, cmap=sns.color_palette("rocket", as_cmap=True), ax=ax)
    ax.set_xlabel("Ratio of selected experts (%)")
    ax.set_ylabel("Layer")
    ax.hlines(range(num_layers), *ax.get_xlim(), linewidth=0.5, color="k", linestyles="dashed")
    ax.set_yticklabels(list(range(1, num_layers + 1, 2)), rotation="horizontal")
    ax.autoscale()
    fig.set_tight_layout(True)
    return fig, saved_args


def main(args):
    configure_logging()
    display_names = args.exp_names if args.display_names is None else args.display_names
    name_dict = {}
    for exp_name, display_name in zip(args.exp_names, display_names):
        name_dict[exp_name] = display_name
    assert len(args.exp_names) == len(display_names)
    accelerator = Accelerator(
        split_batches=True,
        mixed_precision=args.mixed_precision,
        # find_unused_parameters finds unused parameters when using DDP
        # but may affect performance negatively
        # sometimes it happens that somehow it works with this argument, but fails without it
        # so - do not remove
        # see: https://github.com/pytorch/pytorch/issues/43259
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    for exp_name in args.exp_names:
        for exp_id in args.exp_ids:
            run_name = f"{exp_name}_{exp_id}"
            output_run_dir = args.output_dir / f"{name_dict[exp_name]}"
            output_run_dir.mkdir(parents=True, exist_ok=True)
            if accelerator.is_main_process:
                logging.info(f"Processing for run: {run_name}")
            model, run_args, final_results = load_model(args, exp_name, exp_id)
            if run_args.model_class != "dsti_router":
                continue
            del final_results["model_state"]

            moe_executed_experts = setup_and_process(accelerator, args, model, run_args)
            for substructure in ["key", "query", "value", "output", "mlp"]:
                # merge layers by structure
                fig, saved_args = plot_ratio_of_selected_experts_heatmap(moe_executed_experts, substructure)
                save_path = output_run_dir / f"sparsities_{substructure}.png"
                args_save_path = output_run_dir / f"sparsities_{substructure}.pt"
                fig.savefig(save_path)
                plt.close(fig)
                torch.save(saved_args, args_save_path)
                logging.info(f"Figure saved in {str(save_path)}, args saved in {str(args_save_path)}")
