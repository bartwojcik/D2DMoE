import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.dynamic_sparsification.expert_split import train as dsti_expert_split
from methods.dynamic_sparsification.rep_distill_train import train as mha_distill
from methods.dynamic_sparsification.sparse_finetuning import train as sparse_finetune
from methods.dynamic_sparsification.train_routers import train as dsti_train_routers
from train import train
from utils import generate_run_name, submit_job
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.time_vs_plot import get_default_args as get_default_time_plot_args
from visualize.time_vs_plot import main as time_vs_plot


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = 'effbench'

    # account = 'plgccbench-gpu-a100'
    # account = 'plgimpmoe-gpu-a100'
    account = None

    # qos = 'normal'
    qos = 'big'

    # partition = 'plgrid-gpu-a100'
    partition = 'dgxa100'
    # partition = 'dgxa100,dgxh100'
    # partition = 'rtx3080'
    # partition = 'batch'

    timeout = 60 * 24 * 7
    # timeout = 60 * 24 * 2

    gpus_per_task = 1
    gpu_type = ''
    # gpu_type = 'ampere:'
    cpus_per_gpu = 10
    # cpus_per_gpu = 16
    # mem_per_gpu = '64G'
    mem_per_gpu = '32G'
    # mem_per_gpu = None

    executor = submitit.AutoExecutor(folder=os.environ['LOGS_DIR'])
    executor.update_parameters(
        stderr_to_stdout=True,
        timeout_min=timeout,
        slurm_job_name=job_name,
        slurm_account=account,
        slurm_qos=qos,
        slurm_partition=partition,
        slurm_ntasks_per_node=1,
        slurm_cpus_per_gpu=cpus_per_gpu,
        slurm_mem_per_gpu=mem_per_gpu,
    )

    # ════════════════════════ experiment settings ════════════════════════ #

    common_args = get_default_args()
    # exp_ids = [1, 2, 3]
    exp_ids = [1]
    common_args.runs_dir = Path(os.environ['RUNS_DIR'])
    common_args.dataset = 'imagenet'
    common_args.dataset_args = {}
    common_args.dataset_args.variant = 'deit3_rrc'
    common_args.mixup_alpha = 0.8
    common_args.cutmix_alpha = 1.0
    common_args.mixup_smoothing = 0.1
    common_args.batch_size = 128
    common_args.loss_type = 'ce'
    common_args.loss_args = {}
    common_args.optimizer_class = 'adam'
    common_args.optimizer_args = {}
    common_args.optimizer_args.lr = 0.001
    common_args.optimizer_args.weight_decay = 0.0
    common_args.scheduler_class = 'cosine'
    common_args.scheduler_args = {}
    common_args.scheduler_args.eta_min = 1e-6
    common_args.clip_grad_norm = 1.0
    common_args.epochs = 5
    common_args.eval_points = 20
    common_args.use_wandb = True
    common_args.mixed_precision = None

    jobs = []
    run_to_job_map = {}
    exp_names = []
    display_names = []

    # ════════════════════════ base model settings ════════════════════════ #

    base_model_args = deepcopy(common_args)
    base_model_args.model_class = 'tv_vit_b_16'
    base_model_args.model_args = {}
    base_model_args.epochs = 0  # pretrained
    base_model_args.eval_points = 0

    # ════════════════════════ get pretrained base models ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(base_model_args)
        args.exp_id = exp_id
        exp_name, run_name = generate_run_name(args)
        # job = submit_job(executor, train, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
        # jobs.append(job)
        # run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append(f'ViT-B')
    base_exp_name = exp_name

    # ════════════════════════ MHA replacement model settings ════════════════════════ #

    dsti_gpus_per_task = 2
    # dsti_gpus_per_task = 3
    # dsti_gpus_per_task = 4

    distillation_args = deepcopy(common_args)
    distillation_args.model_class = 'mha_rep_distill'
    distillation_args.model_args = {}
    distillation_args.model_args.flops_factor = 1.0
    distillation_args.model_args.mlp_type = 'simple'
    # distillation_args.model_args.mlp_type = 'residual'
    # distillation_args.model_args.dropout = 'same'
    distillation_args.model_args.dropout = 0.05
    # distillation_args.model_args.dropout = 0.0
    distillation_args.base_on = base_exp_name
    distillation_args.epochs = 3
    # distillation_args.epochs = 1
    # distillation_args.epochs = 0
    # distillation_args.epochs = 0.1
    distillation_args.batch_size = 128
    # distillation_args.batch_size = 64
    distillation_args.dsti_distill_loss_type = 'mse'
    distillation_args.eval_points = 0
    distillation_args.optimizer_args.lr = 0.001
    #
    # distillation_args.dsti_enforce_weight = 1e-2
    distillation_args.dsti_enforce_weight = 1e-3
    # distillation_args.dsti_enforce_weight = 8e-4
    # distillation_args.dsti_enforce_weight = 1e-4
    # distillation_args.dsti_enforce_weight = 1e-5
    distillation_args.dsti_enforce_mode = 'relu_hoyer'
    # distillation_args.dsti_enforce_mode = 'relu_l1'
    distillation_args.dsti_enforce_schedule = 'linear'
    # doesn't work with mixed precision yet
    # distillation_args.mixed_precision = 'bf16'

    # ════════════════════════ MHA distillation into non-linear QKVO modules ════════════════════════ #

    base_distill_exp_names = []
    for exp_id in exp_ids:
        args = deepcopy(distillation_args)
        args.exp_id = exp_id
        exp_name, run_name = generate_run_name(args)
        base_run_name = f'{base_exp_name}_{exp_id}'
        if base_run_name in run_to_job_map:
            dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
            executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
        else:
            executor.update_parameters(slurm_additional_parameters={})
        # job = submit_job(executor, mha_distill, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
        # jobs.append(job)
        # run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    base_distill_exp_names.append(exp_names[-1])
    display_names.append(f'MHA distillation')

    # ════════════════════════ sparsity enforcement settings ════════════════════════ #

    dsti_gpus_per_task = 4
    # dsti_gpus_per_task = 3
    # dsti_gpus_per_task = 2

    sparsity_enforcement_args = deepcopy(common_args)
    sparsity_enforcement_args.dsti_enforce_mode = 'relu_hoyer'
    sparsity_enforcement_args.dsti_clamp_displacement = -10.0
    # sparsity_enforcement_args.dsti_enforce_weight = 5e-1
    # this was the previous one
    sparsity_enforcement_args.dsti_enforce_weight = 2e-1
    # sparsity_enforcement_args.dsti_enforce_weight = 1e-1
    # next one to try out
    # sparsity_enforcement_args.dsti_enforce_weight = 5e-2
    # sparsity_enforcement_args.dsti_enforce_weight = 1e-2
    # sparsity_enforcement_args.dsti_enforce_weight = 5e-3
    # sparsity_enforcement_args.dsti_enforce_weight = 1e-3
    # sparsity_enforcement_args.dsti_enforce_weight = 2e-4
    # sparsity_enforcement_args.dsti_enforce_weight = 0.0
    # sparsity_enforcement_args.dsti_enforce_mode = 'relu_l1'
    # sparsity_enforcement_args.dsti_enforce_weight = 1e-3
    # sparsity_enforcement_args.dsti_enforce_mode = 'gelu_preactivations_clamped'
    # sparsity_enforcement_args.dsti_enforce_weight = 0.5
    # sparsity_enforcement_args.dsti_enforce_weight = 0.2
    # sparsity_enforcement_args.dsti_enforce_mode = 'gelu_preactivations_hoyer_clamped'
    # sparsity_enforcement_args.dsti_enforce_weight = 8e-3
    # sparsity_enforcement_args.dsti_enforce_weight = 4e-3
    # sparsity_enforcement_args.dsti_enforce_weight = 1e-3
    #
    # sparsity_enforcement_args.dsti_enforce_schedule = 'linear'
    sparsity_enforcement_args.dsti_enforce_schedule = 'linear_warmup'
    sparsity_enforcement_args.model_class = 'enforce_sparsity'
    sparsity_enforcement_args.model_args = {}
    sparsity_enforcement_args.model_args.apply_to = 'moe_eligible_only'
    sparsity_enforcement_args.epochs = 90
    # sparsity_enforcement_args.epochs = 93
    # sparsity_enforcement_args.epochs = 0.1
    sparsity_enforcement_args.batch_size = 512
    # sparsity_enforcement_args.batch_size = 256
    # sparsity_enforcement_args.eval_points = 91
    sparsity_enforcement_args.eval_points = 46
    # sparsity_enforcement_args.eval_points = 0
    sparsity_enforcement_args.optimizer_args.lr = 2e-5
    # sparsity_enforcement_args.optimizer_args.lr = 2e-4
    # sparsity_enforcement_args.optimizer_args.lr = 5e-4
    # sparsity_enforcement_args.optimizer_args.lr = 4e-3
    sparsity_enforcement_args.optimizer_args.weight_decay = 0.05
    # end LR for scheduler
    sparsity_enforcement_args.scheduler_args.eta_min = 1e-5
    # sparsity_enforcement_args.scheduler_args.eta_min = 1e-5
    # sparsity_enforcement_args.scheduler_args.eta_min = 2e-6
    # no LR scheduler
    # sparsity_enforcement_args.scheduler_class = None
    # sparsity_enforcement_args.optimizer_args.lr = 1e-5
    #
    #
    # sparsity_enforcement_args.dataset_args.variant = 'trivial_augment'
    # sparsity_enforcement_args.mixup_mode = 'elem'
    # sparsity_enforcement_args.mixed_precision = None
    sparsity_enforcement_args.mixed_precision = 'bf16'

    # ════════════════════════ activation sparsity enforcement ════════════════════════ #

    base_enforce_exp_names = []
    # for base_on_exp_name in base_distill_exp_names:
    for base_on_exp_name in [base_exp_name]:
        for exp_id in exp_ids:
            args = deepcopy(sparsity_enforcement_args)
            args.exp_id = exp_id
            args.base_on = base_on_exp_name
            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_on_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, sparse_finetune, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        base_enforce_exp_names.append(exp_names[-1])
        display_names.append(f'Sparsity enforcement')

    # ════════════════════════ dsti moe split settings ════════════════════════ #

    # dsti_gpus_per_task = 4
    # dsti_gpus_per_task = 2
    dsti_gpus_per_task = 1

    expert_split_args = deepcopy(common_args)
    expert_split_args.model_class = 'dsti_expert_split'
    expert_split_args.epochs = 0
    expert_split_args.batch_size = 64
    expert_split_args.model_args = {}

    expert_size = 128
    # expert_size = 64
    # expert_size = 32

    expert_split_args.model_args.expert_size = expert_size
    # expert_split_args.model_args.experts_class = 'execute_all'
    expert_split_args.model_args.experts_class = 'custom_kernel'

    # ════════════════════════ dsti moe split ════════════════════════ #

    base_split_exp_names = []
    for base_on_exp_name in base_enforce_exp_names:
        # for base_on_exp_name in base_relufication_exp_names:
        for exp_id in exp_ids:
            args = deepcopy(expert_split_args)
            args.exp_id = exp_id
            args.base_on = base_on_exp_name
            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_on_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, dsti_expert_split, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        base_split_exp_names.append(exp_names[-1])
        display_names.append(f'D2DMoE expert split')

    # ════════════════════════ dsti router training model settings ════════════════════════ #

    # dsti_gpus_per_task = 4
    # dsti_gpus_per_task = 3
    # dsti_gpus_per_task = 2
    dsti_gpus_per_task = 1

    dsti_routing_args = deepcopy(common_args)
    dsti_routing_args.model_class = 'dsti_router'
    dsti_routing_args.router_loss_type = 'mse'
    dsti_routing_args.epochs = 7
    # dsti_routing_args.epochs = 0.1
    # dsti_routing_args.batch_size = 256
    dsti_routing_args.batch_size = 128
    # dsti_routing_args.batch_size = 64
    dsti_routing_args.optimizer_args.lr = 0.001
    dsti_routing_args.model_args = {}
    dsti_routing_args.model_args.depth = 2
    dsti_routing_args.model_args.width = 128
    # dsti_routing_args.model_args.width = 32
    # dsti_routing_args.model_args.width = 16
    # dsti_routing_args.model_args.activation = 'gelu'
    dsti_routing_args.model_args.activation = 'relu'
    # dsti_routing_args.model_args.activation = 'tanh'
    dsti_routing_args.model_args.output_activation = 'abs'
    # dsti_routing_args.model_args.output_activation = 'relu'
    # dsti_routing_args.model_args.output_activation = 'identity'
    dsti_routing_args.dsti_router_labels_layer = 'output'
    # dsti_routing_args.dsti_router_labels_layer = 'intermediate'
    dsti_routing_args.dsti_router_labels_norm = 2
    dsti_routing_args.dsti_tau_to_eval = [0.5, 0.65, 0.75, 0.85, 0.9, 0.925, 0.95,
                                          0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0]
    dsti_routing_args.dsti_expert_selection_mode = 'dynk_max'
    # dsti_routing_args.dsti_tau_to_eval = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    # dsti_routing_args.dsti_expert_selection_mode = 'dynk'
    dsti_routing_args.eval_points = 4
    # dsti_routing_args.eval_points = 0
    # dsti_routing_args.mixed_precision = None
    dsti_routing_args.mixed_precision = 'bf16'

    # ════════════════════════ dsti router training ════════════════════════ #

    base_routed_dsti_exp_names = []
    for base_on_exp_name in base_split_exp_names:
        for exp_id in exp_ids:
            args = deepcopy(dsti_routing_args)
            args.exp_id = exp_id
            args.base_on = base_on_exp_name
            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_on_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, dsti_train_routers, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        base_routed_dsti_exp_names.append(exp_names[-1])
        display_names.append(f'D2DMoE')

    # ═════════════════════════════════════════════════════════ #

    print(f"Exp names: {exp_names}")
    print(f"Display names: {display_names}")
    print(f"SLURM JIDs: {[job.job_id for job in jobs]}")

    # ════════════════════════ plot cost vs acc ════════════════════════ #

    plot_args = get_default_cost_plot_args()

    out_dir_name = f"vit_{common_args.dataset}_wip_hoyer"
    output_dir = Path(os.environ["RESULTS_DIR"]) / out_dir_name
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = display_names
    plot_args.output_name = "cost_vs"
    plot_args.mode = "acc"
    plot_args.use_wandb = False

    if jobs:
        dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
        executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
    # submit_job(executor, cost_vs_plot, plot_args, gpu_type=gpu_type)

    # ════════════════════════ plot throughput plots - GPU BS 1 ════════════════════════ #
    selected_pairs = [(exp_name, display_name) for exp_name, display_name in zip(exp_names, display_names)
                      if ('_dsti_router_' in exp_name or '_vit_' in exp_name)]
    selected_exp_names, selected_display_names = zip(*selected_pairs)
    # print(f'{selected_pairs=}')
    # print(f'{selected_exp_names=}')
    # print(f'{selected_display_names=}')
    #
    # plot_args = get_default_time_plot_args()
    # plot_args.output_dir = output_dir
    # plot_args.runs_dir = common_args.runs_dir
    # plot_args.exp_names = selected_exp_names
    # plot_args.exp_ids = exp_ids
    # plot_args.display_names = selected_display_names
    # batch_size = 1
    # # plot_args.output_name = f"d2dmoe_bench_cuda_bs_{batch_size}_no_mha"
    # plot_args.mode = "time_vs_flops"
    # plot_args.device = "cuda"
    # plot_args.profile = True
    # plot_args.batch_size = batch_size
    # plot_args.use_wandb = False
    #
    # if jobs:
    #     dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
    #     executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
    # submit_job(executor, time_vs_plot, plot_args, gpu_type=gpu_type)

    # ════════════════════════ plot throughput plots - GPU large BS ════════════════════════ #
    plot_args = get_default_time_plot_args()
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = selected_exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = selected_display_names
    batch_size = 256
    plot_args.output_name = f"d2dmoe_bench_cuda_bs_{batch_size}_es_{expert_size}_no_mha"
    plot_args.mode = "time_vs_flops"
    plot_args.device = "cuda"
    plot_args.profile = True
    plot_args.batch_size = batch_size
    plot_args.use_wandb = False
    if jobs:
        dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
        executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
    submit_job(executor, time_vs_plot, plot_args, gpu_type=gpu_type)

    # same but acc vs wall-clock time - TODO refactor and optimize (this reruns the measurements)
    plot_args.mode = "acc_vs_time"
    if jobs:
        dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
        executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
    partition = 'dgxa100'
    executor.update_parameters(slurm_partition=partition)
    submit_job(executor, time_vs_plot, plot_args, gpu_type=gpu_type)

    # ════════════════════════ plot throughput plots - CPU ════════════════════════ #
    # TODO try out when triton starts supporting CPUs
    # plot_args = get_default_time_plot_args()
    # plot_args.output_dir = output_dir
    # plot_args.runs_dir = common_args.runs_dir
    # plot_args.exp_names = selected_exp_names
    # plot_args.exp_ids = exp_ids
    # plot_args.display_names = selected_display_names
    # batch_size = 1
    # plot_args.output_name = f"d2dmoe_bench_cpu_bs_{batch_size}_es_{expert_size}_no_mha"
    # # plot_args.mode = "time_vs_flops"
    # plot_args.mode = "acc_vs_time"
    # plot_args.device = "cpu"
    # plot_args.profile = False
    # plot_args.batch_size = batch_size
    # plot_args.use_wandb = False
    #
    # cpu_partition = 'dgxh100'
    # cpu_executor = submitit.AutoExecutor(folder=os.environ['LOGS_DIR'])
    # cpu_executor.update_parameters(
    #     stderr_to_stdout=True,
    #     timeout_min=timeout,
    #     slurm_job_name=job_name,
    #     slurm_account=account,
    #     slurm_qos=qos,
    #     slurm_partition=cpu_partition,
    #     slurm_ntasks_per_node=1,
    #     slurm_cpus_per_task=40,
    #     slurm_mem='100G',
    # )
    # if len(jobs) > 0:
    #     dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
    #     cpu_executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
    # submit_job(cpu_executor, time_vs_plot, plot_args, gpu_type=gpu_type)


if __name__ == '__main__':
    main()
