import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from datasets_config import DATASETS_NAME_MAP
from utils import add_save_outputs_hook, configure_logging, find_module_names, get_loader, load_model
from visualize.cost_vs_plot import FONT_SIZE


def get_default_args():
    default_args = OmegaConf.create()
    default_args.runs_dir = Path.cwd() / 'runs'  # Root dir where experiment data was saved.
    default_args.exp_names = []  # Unique experiment names to visualize the results for (excluding exp_id).
    default_args.exp_ids = [0]  # Experiment ids.
    default_args.display_names = None  # Pretty display names that will be used when generating the plot.
    default_args.output_dir = Path.cwd() / 'figures'  # Target directory.
    default_args.batch_size = None  # Batch size.
    default_args.dataset = None  # Dataset to evaluate on. If "None", will use the dataset used by the run.
    default_args.dataset_args = None  # Dataset arguments. If "None", will use same args as in the training run.
    default_args.use_wandb = False  # Use W&B. Will save and load the models from the W&B cloud.
    default_args.mixed_precision = None
    return default_args


def process_dataset(accelerator, model, data_loader, module_outputs):
    globally_sparse = defaultdict(lambda: 0)
    activations_total = defaultdict(lambda: 0)
    token_sparsity = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for x, y in data_loader:
            _ = model(x)
            for k, v in module_outputs.items():
                globally_sparse[k] += (v > 0).int().sum().item()
                activations_total[k] += v.numel()
                token_sparsity[k].append(((v > 0).int().sum(dim=-1) / v.size(-1)).detach().cpu())
    for k in token_sparsity.keys():
        token_sparsity[k] = torch.cat(token_sparsity[k], 0)
    global_activation_stats = {}
    for k in globally_sparse.keys():
        global_activation_stats[k] = globally_sparse[k] / activations_total[k]
    return global_activation_stats, token_sparsity


def setup_and_process(accelerator, args, model, run_args):
    dataset = args.dataset if args.dataset is not None else run_args.dataset
    dataset_args = args.dataset_args if args.dataset_args is not None else run_args.dataset_args
    _, _, eval_data = DATASETS_NAME_MAP[dataset](**dataset_args)
    batch_size = args.batch_size if args.batch_size is not None else run_args.batch_size
    dataloader = get_loader(eval_data, batch_size, accelerator)
    unwrapped_model = accelerator.unwrap_model(model)
    activations = find_module_names(model, lambda _, m: isinstance(m, (torch.nn.ReLU)))
    module_outputs, hook_handles = add_save_outputs_hook(unwrapped_model, activations)
    global_activation_stats, token_sparsity = process_dataset(accelerator,
                                                              model,
                                                              dataloader,
                                                              module_outputs)
    return global_activation_stats, token_sparsity


def plot_sparsity_histogram(token_sparsities, bins=100):
    saved_args = locals()
    token_sparsities = token_sparsities.flatten(0, 1)
    sparsity_histogram, bin_edges = torch.histogram(token_sparsities, bins=bins, range=(0.0, 1.0))
    xs = torch.linspace(0, 1, bins)
    fig, ax = plt.subplots(figsize=(16, 16), facecolor='w', edgecolor='k')
    ax.bar(xs.cpu().numpy(), sparsity_histogram, width=1 / bins)
    # ax.bar(bin_edges.cpu().numpy(), sparsity_histogram.cpu().numpy())
    ax.set_xlabel('Density', fontsize=FONT_SIZE)
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        tick.label.set_fontsize(FONT_SIZE)
    ax.set_ylabel('Number of tokens', fontsize=FONT_SIZE)
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.label.set_fontsize(FONT_SIZE)
    ax.autoscale()
    fig.set_tight_layout(True)
    return fig, saved_args


def plot_subblock_sparsities(layer_sparsity, subblock, title=None):
    subblock_sparsity = [round(v, 4) for k, v in layer_sparsity.items() if subblock in k]
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='w', edgecolor='k')
    ax.bar(range(len(subblock_sparsity)), subblock_sparsity)    
    ax.hlines(np.mean(subblock_sparsity), 0, len(subblock_sparsity), colors='r', linestyles='dashed')
    ax.set_xlabel('Layer', fontsize=FONT_SIZE)
    ax.set_ylabel('Density', fontsize=FONT_SIZE)
    ax.set_xticks(range(len(subblock_sparsity)))
    ax.set_xticklabels(range(len(subblock_sparsity)))
    ax.set_title(title)
    ax.set_ylim(0, 100)
    fig.set_tight_layout(True)
    return fig


def main(args):
    configure_logging()
    display_names = args.exp_names if args.display_names is None else args.display_names
    name_dict = {}
    for exp_name, display_name in zip(args.exp_names, display_names):
        name_dict[exp_name] = display_name
    assert len(args.exp_names) == len(display_names)
    accelerator = Accelerator(split_batches=True, mixed_precision=args.mixed_precision,
                              # find_unused_parameters finds unused parameters when using DDP
                              # but may affect performance negatively
                              # sometimes it happens that somehow it works with this argument, but fails without it
                              # so - do not remove
                              # see: https://github.com/pytorch/pytorch/issues/43259
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
                              )
    for exp_name in args.exp_names:
        for exp_id in args.exp_ids:
            run_name = f'{exp_name}_{exp_id}'
            output_run_dir = args.output_dir / f'{name_dict[exp_name]}'
            output_run_dir.mkdir(parents=True, exist_ok=True)
            if accelerator.is_main_process:
                logging.info(f'Processing for run: {run_name}')
            model, run_args, final_results = load_model(args, exp_name, exp_id)
            # if 'enforce' not in run_args.model_class:
            #     continue
            model = accelerator.prepare(model)
            del final_results['model_state']
            global_activation_stats, token_sparsity = setup_and_process(accelerator, args, model, run_args)
            for k, v in global_activation_stats.items():
                logging.info(f'Global activation sparsity for module {k}: {v * 100}%')
            for key in ['value', 'key', 'query', 'output', 'intermediate_act_fn']:
                fig = plot_subblock_sparsities(global_activation_stats, key, title=key)
                save_path = output_run_dir / f'sparsities_{key}.png'
                fig.savefig(save_path)
                plt.close(fig)
                logging.info(f'Figure saved in {str(save_path)}')
            for k, v in token_sparsity.items():
                fig, saved_args = plot_sparsity_histogram(v)
                save_path = output_run_dir / f'sparsities_{k}.png'
                args_save_path = output_run_dir / f'sparsities_{k}.pt'
                fig.savefig(save_path)
                plt.close(fig)
                torch.save(saved_args, args_save_path)
                logging.info(f'Figure saved in {str(save_path)}, args saved in {str(args_save_path)}')