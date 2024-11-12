import logging
from itertools import cycle
from pathlib import Path

import seaborn
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from architectures.moe.moefication import MoeficationMoE
from common import LOSS_NAME_MAP
from data_utils.data import DATASETS_NAME_MAP
from eval import benchmark_moe, online_evaluate_moe
from utils import find_module_names, get_loader, get_module_by_name, load_model
from visualize.cost_vs_plot import COLORS, FONT_SIZE


def get_default_args():
    default_args = OmegaConf.create()
    default_args.runs_dir = (
            Path.cwd() / "runs"
    )  # Root dir where experiment data was saved.
    default_args.exp_names = (
        []
    )  # Unique experiment names to visualize the results for (excluding exp_id).
    default_args.exp_ids = [0]  # Experiment ids.
    default_args.display_names = (
        None  # Pretty display names that will be used when generating the plot.
    )
    default_args.output_dir = Path.cwd() / "figures"  # Target directory.
    default_args.batch_size = None  # Batch size.
    default_args.dataset = (
        None  # Dataset to evaluate on. If "None", will use the dataset used by the run.
    )
    default_args.dataset_args = (
        None  # Dataset arguments. If "None", will use same args as in the training run.
    )
    default_args.use_wandb = (
        False  # Use W&B. Will save and load the models from the W&B cloud.
    )
    default_args.dsti_tau_to_eval = None  # DSTI taus to evaluate for. If "None", will use same args as in the training run.
    # default_args.k_to_eval = None  # "k" from topk to evaluate. If not "None" will evaluate for ks instead of taus.
    default_args.mixed_precision = None
    return default_args


def set_for_eval_with_tau(model, inspected_moe_name, tau):
    model.eval()
    for m in model.modules():
        if isinstance(m, MoeficationMoE):
            m.forward_mode = "all"
    inspected_moe = get_module_by_name(model, inspected_moe_name)
    inspected_moe.forward_mode = "dynk"
    inspected_moe.tau = tau


def set_for_eval_with_k(model, inspected_moe_name, k):
    model.eval()
    for m in model.modules():
        if isinstance(m, MoeficationMoE):
            m.forward_mode = "all"
    inspected_moe = get_module_by_name(model, inspected_moe_name)
    inspected_moe.forward_mode = "topk"
    if k <= 1.0:
        inspected_moe.k = round(k / inspected_moe.num_experts)
    else:
        inspected_moe.k = k


def get_ks(model, inspected_moe_name):
    inspected_moe = get_module_by_name(model, inspected_moe_name)
    k = inspected_moe.num_experts
    ks = []
    while k > 0:
        ks.append(k)
        k //= 2
    return ks


def plot_tau_curve(tau_to_eval, scores, fractions_of_experts_executed):
    saved_args = locals()
    seaborn.set_style("whitegrid")
    current_palette = cycle(COLORS)
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 9))
    ax2 = ax1.twinx()
    ax1.plot(tau_to_eval.numpy(), scores.numpy(), color=next(current_palette))
    ax2.plot(
        tau_to_eval.numpy(),
        fractions_of_experts_executed.numpy(),
        color=next(current_palette),
    )
    ax1.set_xlabel(f"Tau")
    ax1.set_ylabel(f"Accuracy")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlim(0.0, 1.0)
    ax2.set_ylabel(f"Fraction of executed experts")
    ax2.set_ylim(0.0, 1.0)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(FONT_SIZE - 4)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(FONT_SIZE - 4)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(FONT_SIZE - 4)
    fig.set_tight_layout(True)
    return fig, saved_args


def plot_k_curve(ks_to_eval, scores, fractions_of_experts_executed):
    saved_args = locals()
    seaborn.set_style("whitegrid")
    current_palette = cycle(COLORS)
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 9))
    ax2 = ax1.twinx()
    ax1.plot(ks_to_eval.numpy(), scores.numpy(), color=next(current_palette))
    ax2.plot(
        ks_to_eval.numpy(),
        fractions_of_experts_executed.numpy(),
        color=next(current_palette),
    )
    ax1.set_xlabel(f"k")
    ax1.set_ylabel(f"Accuracy")
    ax1.set_ylim(0.0, 1.0)
    ax2.set_ylabel(f"Fraction of executed experts")
    ax2.set_ylim(0.0, 1.0)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(FONT_SIZE - 4)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(FONT_SIZE - 4)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(FONT_SIZE - 4)
    fig.set_tight_layout(True)
    return fig, saved_args


def main(args):
    logging.basicConfig(
        format=(
            "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] "
            "%(message)s"
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
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
            if accelerator.is_main_process:
                logging.info(f"Processing for run: {run_name}")
            model, run_args, final_results = load_model(args, exp_name, exp_id)
            if run_args.model_class != "dsti_router":
                continue
            model = accelerator.prepare(model)
            del final_results["model_state"]
            dataset = args.dataset if args.dataset is not None else run_args.dataset
            dataset_args = (
                args.dataset_args
                if args.dataset_args is not None
                else run_args.dataset_args
            )
            _, _, eval_data = DATASETS_NAME_MAP[dataset](**dataset_args)
            batch_size = (
                args.batch_size if args.batch_size is not None else run_args.batch_size
            )
            dataloader = get_loader(eval_data, batch_size, accelerator)
            unwrapped_model = accelerator.unwrap_model(model)
            cost_without_experts, token_expert_costs, model_params = benchmark_moe(
                unwrapped_model, dataloader
            )
            tau_to_eval = (
                args.dsti_tau_to_eval
                if args.dsti_tau_to_eval is not None
                else run_args.dsti_tau_to_eval
            )
            criterion_type = LOSS_NAME_MAP[run_args.loss_type]
            moe_module_names = find_module_names(
                model, lambda _, m: isinstance(m, MoeficationMoE)
            )
            for moe_module_name in moe_module_names:
                if accelerator.is_main_process:
                    logging.info(
                        f"Processing for run: {run_name} module: {moe_module_name}"
                    )
                tau_scores = []
                # tau_losses = []
                tau_fractions_of_experts_executed = []
                for tau in tau_to_eval:
                    if accelerator.is_main_process:
                        logging.info(f"Evaluating for tau={tau}")
                    set_for_eval_with_tau(model, moe_module_name, tau)
                    _, test_acc, _, _, executed_expert_tokens, total_expert_tokens = (
                        online_evaluate_moe(
                            accelerator,
                            model,
                            dataloader,
                            criterion_type,
                            cost_without_experts,
                            token_expert_costs,
                            return_counts=True,
                        )
                    )
                    fraction_of_experts_executed = (
                            executed_expert_tokens[moe_module_name]
                            / total_expert_tokens[moe_module_name]
                    )
                    tau_scores.append(test_acc)
                    tau_fractions_of_experts_executed.append(
                        fraction_of_experts_executed
                    )
                    # tau_losses.append(test_loss)
                if accelerator.is_main_process:
                    fig, saved_args = plot_tau_curve(
                        torch.tensor(tau_to_eval),
                        torch.tensor(tau_scores),
                        torch.tensor(tau_fractions_of_experts_executed),
                    )
                    # TODO add mean+std for multiple seeds
                    output_run_dir = args.output_dir / f"{name_dict[exp_name]}"
                    output_run_dir.mkdir(parents=True, exist_ok=True)
                    save_path = output_run_dir / f"taus_for_{moe_module_name}.png"
                    args_save_path = output_run_dir / f"taus_for_{moe_module_name}.pt"
                    fig.savefig(save_path)
                    plt.close(fig)
                    torch.save(saved_args, args_save_path)
                    logging.info(
                        f"Figure saved in {str(save_path)}, args saved in {str(args_save_path)}"
                    )
                k_scores = []
                k_fractions_of_experts_executed = []
                ks = get_ks(model, moe_module_name)
                for k in ks:
                    if accelerator.is_main_process:
                        logging.info(f"Evaluating for k={k}")
                    set_for_eval_with_k(model, moe_module_name, k)
                    _, test_acc, _, _, executed_expert_tokens, total_expert_tokens = (
                        online_evaluate_moe(
                            accelerator,
                            model,
                            dataloader,
                            criterion_type,
                            cost_without_experts,
                            token_expert_costs,
                            return_counts=True,
                        )
                    )
                    fraction_of_experts_executed = (
                            executed_expert_tokens[moe_module_name]
                            / total_expert_tokens[moe_module_name]
                    )
                    k_scores.append(test_acc)
                    k_fractions_of_experts_executed.append(fraction_of_experts_executed)
                if accelerator.is_main_process:
                    fig, saved_args = plot_k_curve(
                        torch.tensor(ks),
                        torch.tensor(k_scores),
                        torch.tensor(k_fractions_of_experts_executed),
                    )
                    # TODO add mean+std for multiple seeds
                    output_run_dir = args.output_dir / f"{name_dict[exp_name]}"
                    output_run_dir.mkdir(parents=True, exist_ok=True)
                    save_path = output_run_dir / f"ks_for_{moe_module_name}.png"
                    args_save_path = output_run_dir / f"ks_for_{moe_module_name}.pt"
                    fig.savefig(save_path)
                    plt.close(fig)
                    torch.save(saved_args, args_save_path)
                    logging.info(
                        f"Figure saved in {str(save_path)}, args saved in {str(args_save_path)}"
                    )
