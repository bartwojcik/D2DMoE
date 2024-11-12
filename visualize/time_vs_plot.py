import logging
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch._C._profiler import ProfilerActivity
from torch.profiler import profile

from architectures.custom import simplify_mha
from architectures.moe.moe_layers import CustomKernelExperts
from architectures.moe.moefication import MoeficationMoE
from common import LOSS_NAME_MAP
from data_utils.data import DATASETS_NAME_MAP
from eval import evaluate_model_throughput
from utils import get_plain_loader, load_model
from visualize.cost_vs_plot import compute_means_and_stds, plot_score_eff_tradeoff


def get_default_args():
    default_args = OmegaConf.create()
    default_args.runs_dir = Path.cwd() / 'runs'  # Root dir where experiment data was saved.
    default_args.exp_names = []  # Unique experiment names to visualize the results for (excluding exp_id).
    default_args.exp_ids = [0]  # Experiment ids.
    default_args.display_names = None  # Pretty display names that will be used when generating the plot.
    default_args.output_dir = Path.cwd() / 'figures'  # Target directory.
    default_args.output_name = 'time_vs'  # Output file name prefix to use.
    default_args.mode = 'time'  # Type of plot to generate.
    default_args.device = 'cpu'  # Device to use for benchmarking.
    default_args.batch_size = None  # Batch size.
    default_args.dataset = None  # Dataset to evaluate on. If "None", will use the dataset used by the run.
    default_args.dataset_args = None  # Dataset arguments. If "None", will use same args as in the training run.
    default_args.use_wandb = False  # Use W&B. Will save and load the models from the W&B cloud.
    default_args.profile = False  # Whether to profile the forward pass with torch.profiler module.
    return default_args


def set_slow_backend(_args, model, run_args):
    # TODO this needs to be updated to work for new models
    if 'vit' in run_args.model_class:
        simplify_mha(model)
    elif 'dsti' in run_args.model_class:
        simplify_mha(model)
        mode = 'masked'
        for m in model.modules():
            if isinstance(m, CustomKernelExperts):
                m.forward_mode = mode


def set_performance_backend(args, model, run_args):
    if 'dsti' in run_args.model_class:
        # TODO always choose the same implementation when triton starts supporting CPUs
        if 'cuda' in args.device:
            mode = 'triton_atomic'
        else:
            mode = 'naive'
        for name, m in model.named_modules():
            if isinstance(m, CustomKernelExperts):
                logging.info(f'Setting mode "{mode}" for module {name}')
                m.forward_mode = mode


def set_for_eval_with_dynk(model, tau, mode=None):
    model.eval()
    for m in model.modules():
        if isinstance(m, MoeficationMoE):
            m.forward_mode = 'dynk_max' if mode is None else mode
            m.tau = tau


def profile_model(model, dataloader, name, output_dir, device, batch_size):
    logging.info(f'Profiling {name}')
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    def trace_handler(p):
        log_file_path = output_dir / f'{name}_bs_{batch_size}_d_{device}_prof_{p.step_num}.log'
        trace_file_path = output_dir / f'{name}_bs_{batch_size}_d_{device}_prof_{p.step_num}.json'
        with open(log_file_path, 'w') as log_f:
            print(p.key_averages().table(sort_by='self_cuda_time_total', row_limit=15), file=log_f)
            print(p.key_averages().table(sort_by='cpu_memory_usage', row_limit=15), file=log_f)
        p.export_chrome_trace(str(trace_file_path))
        logging.info(f'Saving profiling results in {str(trace_file_path)}')

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # with_stack=True,
            record_shapes=True,
            profile_memory=True,
            schedule=torch.profiler.schedule(
                skip_first=5,
                wait=2,
                warmup=2,
                active=1,
                repeat=1),
            on_trace_ready=trace_handler
    ) as p:
        for X, y in dataloader:
            with torch.inference_mode():
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                _y_pred = model(X)
            if not isinstance(device, str) or 'cpu' not in device:
                torch.cuda.synchronize(device=device)
            p.step()


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
    for exp_name in args.exp_names:
        for exp_id in args.exp_ids:
            run_name = f'{exp_name}_{exp_id}'
            logging.info(f'Processing for: {run_name} ({args.mode})')
            model, run_args, final_results = load_model(args, exp_name, exp_id)
            model.to(args.device)
            del final_results['model_state']
            logging.info(f'Preparing dataloader')
            dataset = args.dataset if args.dataset is not None else run_args.dataset
            dataset_args = args.dataset_args if args.dataset_args is not None else run_args.dataset_args
            _, _, data = DATASETS_NAME_MAP[dataset](**dataset_args)
            batch_size = args.batch_size if args.batch_size is not None else run_args.batch_size
            dataloader = get_plain_loader(data, batch_size)
            criterion_type = LOSS_NAME_MAP[run_args.loss_type]
            logging.info(f'Setting appropriate backend')
            set_performance_backend(args, model, run_args)
            expert_selection_mode = final_results['args']['dsti_expert_selection_mode']
            # optionally verify if the selected backend works on a batch of data
            # verify_performance_backend(args, run_args, model, dataloader, device=args.device)
            # profiling
            if args.profile:
                if 'dsti' in run_args.model_class:
                    set_for_eval_with_dynk(model, 1.0, expert_selection_mode)
                profile_model(model, dataloader, run_name, args.output_dir, args.device, args.batch_size)
            #
            if 'model_flops' in final_results:
                logging.info(f'Calling "evaluate_model_throughput"')
                loss, acc, throughput, duration, dataset_size = evaluate_model_throughput(model, dataloader,
                                                                                          criterion_type,
                                                                                          device=args.device, )
                if args.mode == 'throughput_vs_flops':
                    # hacky - ignore the dict keys
                    # TODO change the key names to more abstract ones
                    final_results['final_score'] = throughput
                    final_results['final_alt_score'] = acc
                elif args.mode == 'time_vs_flops':
                    final_results['final_score'] = duration / dataset_size
                    final_results['final_alt_score'] = acc
                elif args.mode == 'acc_vs_throughput':
                    final_results['final_score'] = acc
                    final_results['model_flops'] = throughput
                elif args.mode == 'acc_vs_time':
                    final_results['final_score'] = acc
                    final_results['model_flops'] = duration / dataset_size
                else:
                    raise ValueError(f'Illegal args.mode value: {args.mode}')
                core_stats[run_name] = final_results
            elif 'dsti' in run_args.model_class:
                final_scores, final_flops, final_alt_scores = [], [], []
                for tau_to_eval in final_results['args']['dsti_tau_to_eval']:
                    logging.info(f'Calling "evaluate_model_throughput" for tau={tau_to_eval}')
                    set_for_eval_with_dynk(model, tau_to_eval, expert_selection_mode)
                    loss, acc, throughput, duration, dataset_size = evaluate_model_throughput(model, dataloader,
                                                                                              criterion_type,
                                                                                              device=args.device, )
                    if args.mode == 'throughput_vs_flops':
                        # hacky - ignore the dict keys
                        final_scores.append(throughput)
                        final_alt_scores.append(acc)
                    elif args.mode == 'time_vs_flops':
                        final_scores.append(duration / dataset_size)
                        final_alt_scores.append(acc)
                    elif args.mode == 'acc_vs_throughput':
                        final_scores.append(acc)
                        final_flops.append(throughput)
                    elif args.mode == 'acc_vs_time':
                        final_scores.append(acc)
                        final_flops.append(duration / dataset_size)
                    # elif args.mode == 'latency':
                    #     ...
                    else:
                        raise ValueError(f'Illegal args.mode value: {args.mode}')
                final_results['final_scores'] = final_scores
                final_results['final_flops'] = final_flops if final_flops else final_results['final_flops']
                if final_alt_scores:
                    final_results['final_alt_scores'] = final_alt_scores
                point_stats[run_name] = final_results
    logging.info(f'Measurements completed')
    core_stats, point_stats, ee_stats = compute_means_and_stds(args.exp_names, args.exp_ids, core_stats, point_stats,
                                                               ee_stats)
    for exp_name, display_name in zip(args.exp_names, display_names):
        name_dict[exp_name] = display_name
    # TODO ifology refactor
    if args.mode == 'acc_vs_throughput':
        x_label = 'Throughput (samples/s)'
    elif args.mode == 'acc_vs_time':
        x_label = 'Wall-clock time (s)'
    else:
        x_label = 'FLOPs'
    if args.mode in ['acc_vs_throughput', 'acc_vs_time']:
        y_label = 'Accuracy'
    elif args.mode == 'time_vs_flops':
        y_label = 'Wall-clock time (s)'
    else:
        y_label = 'Throughput (samples/s)'
    fig, saved_args = plot_score_eff_tradeoff(core_stats, point_stats, ee_stats, name_dict,
                                              x_label=x_label,
                                              y_label=y_label)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f'{args.output_name}_{args.mode}.png'
    args_save_path = args.output_dir / f'{args.output_name}_{args.mode}.pt'
    fig.savefig(save_path)
    plt.close(fig)
    torch.save(saved_args, args_save_path)
    logging.info(f'Figure saved in {str(save_path)}, args saved in {str(args_save_path)}')


if __name__ == "__main__":
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    main(args)
