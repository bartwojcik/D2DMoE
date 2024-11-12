import logging
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Dict, List

import matplotlib
import seaborn as seaborn
import torch
from accelerate import Accelerator
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from common import LOSS_NAME_MAP
from data_utils.data import DATASETS_NAME_MAP
from eval import (
    evaluate_calibration,
    evaluate_earlyexiting_calibration,
    evaluate_earlyexiting_ood_detection,
    evaluate_ood_detection,
    get_preds,
    get_preds_earlyexiting,
    test_classification,
)
from utils import get_loader, load_model, retrieve_final
from visualize import mean_std


# TODO refactor "ifology" code throughout this file, think about abstracting out "loss" vs "acc"


def get_default_args():
    default_args = OmegaConf.create()
    default_args.runs_dir = Path.cwd() / 'runs'  # Root dir where experiment data was saved.
    default_args.exp_names = []  # Unique experiment names to visualize the results for (excluding exp_id).
    default_args.exp_ids = [0]  # Experiment ids.
    default_args.display_names = None  # Pretty display names that will be used when generating the plot.
    default_args.output_dir = Path.cwd() / 'figures'  # Target directory.
    default_args.output_name = 'cost_vs'  # Output file name prefix to use.
    default_args.mode = 'acc'  # Type of plot to generate. Choices: ['acc', 'calibration', 'ood_detection']
    default_args.batch_size = None  # Batch size.
    default_args.dataset = None  # Dataset to evaluate on. If "None", will use the dataset used by the run.
    default_args.dataset_args = None  # Dataset arguments. If "None", will use same args as in the training run.
    default_args.ood_dataset = None  # Out-of-Distribution dataset.
    default_args.use_wandb = False  # Use W&B. Will save and load the models from the W&B cloud.
    return default_args


PRETTY_NAME_DICT = {
    "acc": "Accuracy",
    "loss": "Loss",
    "calibration": "Calibration Error",
    "ood_detection": "OOD detection AUROC",
}

# generic
FONT_SIZE = 22
COLORS = ['#000000', '#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#810f7c', '#0ee5a8', '#f04f6d', '#00ffff', '#0000ff']


def mark_single_result(
        stats: Dict,
        ax: matplotlib.axes.SubplotBase,
        name: str,
        color: str,
        marker: str,
        mode: str = "acc",
):
    cost = stats["model_flops"].numpy()
    if mode == "acc":
        key = "final_score"
    elif mode == "loss":
        key = "final_loss"
    else:
        raise NotImplementedError()
    score_mean = stats[key].numpy()
    if marker == "lines":
        # only one line may be specified; full height
        # ax.axvline(x=cost, color=color, linestyle='--', label=name)
        ax.axvline(x=cost, color=color, linestyle="--")
        ax.axhline(y=score_mean, color=color, linestyle="--")
    else:
        ax.scatter(
            cost,
            score_mean,
            marker=marker,
            label=name,
            color=color,
            s=125,
            zorder=3,
            linewidths=1.0,
        )
        if mode == "acc":
            std_key = "final_score_std"
        elif mode == "loss":
            std_key = "final_loss_std"
        else:
            raise NotImplementedError()
        if std_key in stats:
            x_std = stats["model_flops_std"].numpy()
            score_std = stats[std_key].numpy()
            ax.errorbar(
                cost, score_mean, xerr=x_std, yerr=score_std, ecolor=color, alpha=0.5
            )


def draw_for_ics(stats: Dict, ax: matplotlib.axes.SubplotBase, name: str, color: str, marker: str):
    costs = stats['head_flops'].numpy()
    scores = stats['head_scores'].numpy()
    ax.scatter(costs, scores, marker=marker, color=color, s=90, zorder=3, edgecolors='black', linewidths=1.0)
    if 'head_scores_std' in stats:
        x_std = stats['head_flops_std'].numpy()
        head_stds = stats['head_scores_std'].numpy()
        ax.errorbar(costs, scores, xerr=x_std, yerr=head_stds, ecolor=color, fmt=' ', alpha=0.5)


def draw_for_points(
        stats: Dict,
        ax: matplotlib.axes.SubplotBase,
        name: str,
        color: str,
        marker: str,
        mode: str = "acc",
):
    costs = stats["final_flops"].numpy()
    if mode == "acc":
        scores = stats["final_scores"].numpy()
    elif mode == "loss":
        scores = stats["test_losses"].numpy()
    else:
        raise NotImplementedError()
    ax.scatter(
        costs,
        scores,
        marker=marker,
        label=f"{name}",
        color=color,
        s=125,
        zorder=3,
        linewidths=1.0,
    )

    if mode == "acc":
        key = "final_scores_std"
    elif mode == "loss":
        key = "test_losses_std"
    else:
        raise NotImplementedError()

    if key in stats:
        x_std = stats["final_flops_std"].numpy()
        score_stds = stats[key].numpy()
        ax.errorbar(costs, scores, xerr=x_std, yerr=score_stds, color=color, alpha=0.5)


def draw_for_ics(
        stats: Dict, ax: matplotlib.axes.SubplotBase, name: str, color: str, marker: str
):
    costs = stats["head_flops"].numpy()
    scores = stats["head_scores"].numpy()
    ax.scatter(
        costs,
        scores,
        marker=marker,
        color=color,
        s=90,
        zorder=3,
        edgecolors="black",
        linewidths=1.0,
    )
    if "head_scores_std" in stats:
        x_std = stats["head_flops_std"].numpy()
        head_stds = stats["head_scores_std"].numpy()
        ax.errorbar(
            costs, scores, xerr=x_std, yerr=head_stds, ecolor=color, fmt=" ", alpha=0.5
        )


def draw_for_thresholds(
        stats: Dict, ax: matplotlib.axes.SubplotBase, name: str, color: str
):
    # thresholds = stats['thresholds'].numpy()
    costs = stats['threshold_flops'].numpy()
    scores = stats['threshold_scores'].numpy()
    ax.plot(costs, scores, label=name, color=color)
    if 'threshold_scores_std' in stats:
        scores_stds = stats['threshold_scores_std'].numpy()
        ax.fill_between(costs, scores - scores_stds, scores + scores_stds, alpha=0.3, color=color)


def plot_score_eff_tradeoff(
        core_stats: Dict,
        point_stats: Dict,
        ee_stats: Dict,
        name_dict: Dict[str, str],
        x_label: str = "Inference FLOPs",
        y_label: str = None,
        title: str = None,
        mode: str = "acc",
):
    saved_args = locals()
    seaborn.set_style("whitegrid")
    current_palette = cycle(COLORS)
    colors = {}
    markers = {}
    join_data = defaultdict(list)
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    for i, (run_name, display_name) in enumerate(name_dict.items()):
        colors[run_name] = next(current_palette)
        if run_name in core_stats:
            mark_single_result(
                core_stats[run_name], ax, display_name, colors[run_name], "X", mode=mode
            )
        if run_name in point_stats:
            draw_for_points(
                point_stats[run_name],
                ax,
                name_dict[run_name],
                colors[run_name],
                "*",
                mode=mode,
            )
        if run_name in ee_stats:
            draw_for_ics(ee_stats[run_name], ax, name_dict[run_name], colors[run_name], '.')
            draw_for_thresholds(ee_stats[run_name], ax, name_dict[run_name], colors[run_name])
            last_ee_index = i
    # ax.legend(loc='upper left', prop={'size': FONT_SIZE - 4})
    # ax.legend(loc='lower right', prop={'size': FONT_SIZE - 4})
    ax.legend(loc='best', prop={'size': FONT_SIZE - 4})
    ax.set_title(title, fontdict={'fontsize': FONT_SIZE + 1})
    ax.set_xlabel(x_label, fontsize=FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE)
    # ax.set_xlim(right=1.1 * baseline_ops)
    assert len(core_stats) > 0 or len(point_stats) > 0 or len(ee_stats) > 0
    base_model_run_name = next(iter(core_stats.keys()))
    # baseline_ops = core_stats[base_model_run_name]['model_flops'].numpy()
    # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(baseline_ops / 4))
    # ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=baseline_ops))
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 4)
    ax.tick_params(axis='both', which='minor', labelsize=FONT_SIZE - 6)
    # labels - add pretty "ICs"
    # handles, labels = ax.get_legend_handles_labels()
    # circles = []
    # for run_name, _ in ee_stats.items():
    #     circles += [matplotlib.lines.Line2D([], [], color=colors[run_name], marker='o', linestyle='None',
    #                                         markersize=10)]
    # if len(ee_stats) > 0:
    #     handles.insert(last_ee_index + 1, tuple(circles))
    #     labels.insert(last_ee_index + 1, 'EE heads')
    #     # handles += []
    #     # labels += ['EE heads']
    #     ax.legend(handles=handles, labels=labels, prop={'size': FONT_SIZE},
    #               handler_map={tuple: HandlerTuple(ndivide=None)})
    fig.set_tight_layout(True)
    return fig, saved_args


def compute_means_and_stds(
        exp_names: List[str],
        exp_ids: List[int],
        core_stats: Dict,
        point_stats: Dict,
        ee_stats: Dict,
        mode: str = "acc",
):
    processed_core_stats = {}
    processed_point_stats = {}
    processed_ee_stats = {}
    if mode == "acc":
        y_key_core = "final_score"
        y_key_point = "final_scores"
        y_key_ee_heads = "head_scores"
        y_key_thresholds = "threshold_scores"
    elif mode == "loss":
        y_key_core = "final_loss"
        y_key_point = "test_losses"
        y_key_ee_heads = "head_losses"
        y_key_thresholds = "threshold_losses"

    mean_std(
        exp_names, exp_ids, core_stats, processed_core_stats, "model_flops", y_key_core
    )
    mean_std(
        exp_names,
        exp_ids,
        point_stats,
        processed_point_stats,
        "final_flops",
        y_key_point,
    )
    mean_std(
        exp_names, exp_ids, ee_stats, processed_ee_stats, "head_flops", y_key_ee_heads
    )
    mean_std(
        exp_names,
        exp_ids,
        ee_stats,
        processed_ee_stats,
        "threshold_flops",
        y_key_thresholds,
    )
    # copy_entry(exp_names, exp_ids, ee_stats, processed_ee_stats, 'thresholds')
    return processed_core_stats, processed_point_stats, processed_ee_stats


def main(args):
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    core_stats = {}
    ee_stats = {}
    point_stats = {}
    name_dict = {}
    display_names = args.exp_names if args.display_names is None else args.display_names
    assert len(args.exp_names) == len(display_names)
    accelerator = Accelerator(split_batches=True)
    for exp_name in args.exp_names:
        for exp_id in args.exp_ids:
            run_name = f'{exp_name}_{exp_id}'
            logging.info(f'Processing for: {run_name} ({args.mode})')
            # TODO this mess really needs refactoring :)
            # TODO possibly split this into two scripts: one that generates outputs for both datasets and one that plots the data
            if args.mode in ("acc", "loss"):
                if args.dataset is None:
                    final_results = retrieve_final(args, run_name)
                    del final_results['model_state']
                    if 'thresholds' in final_results:
                        ee_stats[run_name] = final_results
                    elif 'hyperparam_values' in final_results:
                        point_stats[run_name] = final_results
                    elif 'final_score' in final_results:
                        core_stats[run_name] = final_results
                    else:
                        logging.info(f'Skipping {run_name} as it has not any recognizable data to plot')
            elif args.mode == 'calibration':
                model, run_args, final_results = load_model(args, exp_name, exp_id)
                model = accelerator.prepare(model)
                del final_results['model_state']
                dataset = args.dataset if args.dataset is not None else run_args.dataset
                dataset_args = args.dataset_args if args.dataset_args is not None else run_args.dataset_args
                _, _, id_data = DATASETS_NAME_MAP[dataset](**dataset_args)
                batch_size = args.batch_size if args.batch_size is not None else run_args.batch_size
                id_dataloader = get_loader(id_data, batch_size, accelerator)
                if 'thresholds' in final_results:
                    id_preds, id_labels = get_preds_earlyexiting(accelerator, model, id_dataloader)
                    final_results.update(
                        evaluate_earlyexiting_calibration(id_preds, id_labels,
                                                          final_results['head_flops'],
                                                          final_results['thresholds']))
                    ee_stats[run_name] = final_results
                elif 'hyperparam_values' in final_results:
                    raise NotImplementedError('TODO')
                else:
                    id_preds, id_labels = get_preds(accelerator, model, id_dataloader)
                    final_results.update(evaluate_calibration(id_preds, id_labels))
                    core_stats[run_name] = final_results
                    # TODO update FLOPs for non-static models
            elif args.mode == 'ood_detection':
                model, run_args, final_results = load_model(args, exp_name, exp_id)
                model = accelerator.prepare(model)
                del final_results['model_state']
                batch_size = args.batch_size if args.batch_size is not None else run_args.batch_size
                dataset = args.dataset if args.dataset is not None else run_args.dataset
                dataset_args = args.dataset_args if args.dataset_args is not None else run_args.dataset_args
                _, _, id_data = DATASETS_NAME_MAP[dataset](**dataset_args)
                id_dataloader = get_loader(id_data, batch_size, accelerator)
                _, _, ood_data = DATASETS_NAME_MAP[args.ood_dataset](**run_args.dataset_args)
                ood_dataloader = get_loader(ood_data, batch_size, accelerator)
                if 'thresholds' in final_results:
                    id_preds, _ = get_preds_earlyexiting(accelerator, model, id_dataloader)
                    ood_preds, _ = get_preds_earlyexiting(accelerator, model, ood_dataloader)
                    final_results.update(
                        evaluate_earlyexiting_ood_detection(id_preds, ood_preds, final_results['head_flops'],
                                                            final_results['thresholds']))
                    ee_stats[run_name] = final_results
                elif 'hyperparam_values' in final_results:
                    raise NotImplementedError('TODO')
                else:
                    id_preds, _ = get_preds(accelerator, model, id_dataloader)
                    ood_preds, _ = get_preds(accelerator, model, ood_dataloader)
                    final_results.update(evaluate_ood_detection(id_preds, ood_preds))
                    core_stats[run_name] = final_results
                    # TODO update FLOPs for non-static models
    core_stats, point_stats, ee_stats = compute_means_and_stds(
        args.exp_names, args.exp_ids, core_stats, point_stats, ee_stats, mode=args.mode
    )
    for exp_name, display_name in zip(args.exp_names, display_names):
        name_dict[exp_name] = display_name
    fig, saved_args = plot_score_eff_tradeoff(
        core_stats,
        point_stats,
        ee_stats,
        name_dict,
        y_label=PRETTY_NAME_DICT[args.mode],
        mode=args.mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f'{args.output_name}_{args.mode}.pdf'
    args_save_path = args.output_dir / f'{args.output_name}_{args.mode}.pt'
    fig.savefig(save_path)
    plt.close(fig)
    torch.save(saved_args, args_save_path)
    logging.info(f'Figure saved in {str(save_path)}, args saved in {str(args_save_path)}')


if __name__ == "__main__":
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    main(args)
