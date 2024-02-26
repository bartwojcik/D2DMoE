import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from train import train
from methods.early_exit import train as train_ee
from methods.moefication.relufication import train as relufication
from methods.moefication.expert_split import train as split_moe
from methods.moefication.train_routers import train as train_routers
from methods.dynamic_sparsification.rep_distill_train import train as mha_distill
from methods.dynamic_sparsification.sparse_finetuning import train as sparse_finetune
from methods.dynamic_sparsification.expert_split import train as dsti_expert_split
from methods.dynamic_sparsification.train_routers import train as dsti_train_routers
from utils import generate_run_name, submit_job
from visualize.activation_sparsity import get_default_args as get_default_activation_sparsity_args
from visualize.activation_sparsity import main as activation_sparsity_plot
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot
from visualize.dsti_one_at_a_time import main as dsti_one_at_a_time_plot
from visualize.dsti_one_at_a_time import get_default_args as get_default_dsti_one_at_a_time_args
from visualize.moe_image_spatial_load import get_default_args as get_default_moe_image_spatial_load_args
from visualize.moe_image_spatial_load import main as dsti_spatial_load_plot


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = 'effbench'

    account = None

    # qos = 'normal'
    qos = 'big'

    partition = 'dgxa100'

    timeout = 60 * 24 * 7
    # timeout = 60 * 24 * 2

    gpus_per_task = 1
    gpu_type = ''
    # gpu_type = 'ampere:'
    cpus_per_gpu = 10
    # cpus_per_gpu = 16
    mem_per_gpu = '64G'
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
        slurm_gpus_per_task=f'{gpu_type}{gpus_per_task}',
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
        # job = submit_job(executor, train, args, num_gpus=gpus_per_task)
        # jobs.append(job)
        # run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append(f'ViT-B')
    base_exp_name = exp_name

    # ════════════════════════ ZTW model settings ════════════════════════ #

    ee_gpus_per_task = 2
    # ee_gpus_per_task = 4

    cascading_model_args = deepcopy(common_args)
    cascading_model_args.epochs = 95
    cascading_model_args.batch_size = 256
    cascading_model_args.with_backbone = False
    cascading_model_args.model_class = 'ztw_cascading'
    cascading_model_args.model_args = {}
    cascading_model_args.model_args.head_type = 'vit_cascading_3l_head'
    cascading_model_args.model_args.place_at = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cascading_model_args.optimizer_args = {}
    cascading_model_args.optimizer_args.lr = 0.001
    cascading_model_args.optimizer_args.weight_decay = 0.00005
    cascading_model_args.mixed_precision = 'bf16'

    ensembling_model_args = deepcopy(common_args)
    ensembling_model_args.with_backbone = False
    ensembling_model_args.batch_size = 256
    ensembling_model_args.epochs = 5
    ensembling_model_args.model_class = 'ztw_ensembling'
    ensembling_model_args.model_args = {}
    ensembling_model_args.optimizer_class = 'adam'
    ensembling_model_args.optimizer_args = {}
    ensembling_model_args.optimizer_args.lr = 0.001
    ensembling_model_args.optimizer_args.weight_decay = 0.0

    # ════════════════════════ train ZTW model for comparison ════════════════════════ #

    executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}{ee_gpus_per_task}')
    for exp_id in exp_ids:
        args = deepcopy(cascading_model_args)
        args.exp_id = exp_id
        args.base_on = base_exp_name  # path to the base model will be inferred from the experiment name
        base_run_name = f'{base_exp_name}_{exp_id}'
        if base_run_name in run_to_job_map:
            dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
            executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
        else:
            executor.update_parameters(slurm_additional_parameters={})
        job = submit_job(executor, train_ee, args, num_gpus=ee_gpus_per_task)
        jobs.append(job)
        exp_name, run_name = generate_run_name(args)
        run_to_job_map[run_name] = job
    cascading_exp_name = exp_name
    for exp_id in exp_ids:
        args = deepcopy(ensembling_model_args)
        args.exp_id = exp_id
        args.base_on = cascading_exp_name
        exp_name, run_name = generate_run_name(args)
        base_run_name = f'{cascading_exp_name}_{exp_id}'
        if base_run_name in run_to_job_map:
            dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
            executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
        else:
            executor.update_parameters(slurm_additional_parameters={})
        # job = submit_job(executor, train_ee, args, num_gpus=ee_gpus_per_task)
        # jobs.append(job)
        # run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append(f'ZTW')


    # ════════════════════════ relufication model settings ════════════════════════ #

    moefication_gpus_per_task = 2
    executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}{moefication_gpus_per_task}')

    relufication_args = deepcopy(common_args)
    relufication_args.model_class = 'relufication'
    relufication_args.model_args = {}
    relufication_args.model_args.mode = 'ffns_only'
    relufication_args.base_on = base_exp_name
    relufication_args.epochs = 90
    # relufication_args.epochs = 0.1
    relufication_args.batch_size = 256
    # relufication_args.batch_size = 128
    relufication_args.eval_points = 91
    # relufication_args.eval_points = 0
    relufication_args.optimizer_args.lr = 0.0001
    relufication_args.mixed_precision = 'bf16'

    # ════════════════════════ relufication ════════════════════════ #

    base_relufication_exp_names = []
    for exp_id in exp_ids:
        args = deepcopy(relufication_args)
        args.exp_id = exp_id
        exp_name, run_name = generate_run_name(args)
        base_run_name = f'{base_exp_name}_{exp_id}'
        if base_run_name in run_to_job_map:
            dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
            executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
        else:
            executor.update_parameters(slurm_additional_parameters={})
        # job = submit_job(executor, relufication, args, num_gpus=moefication_gpus_per_task)
        # jobs.append(job)
        # run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    base_relufication_exp_names.append(exp_names[-1])
    display_names.append(f'ReLUfication')

    # ════════════════════════ moe split settings ════════════════════════ #

    moefication_gpus_per_task = 1
    executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}{moefication_gpus_per_task}')

    moefication_split_args = deepcopy(common_args)
    moefication_split_args.model_class = 'moefication'
    moefication_split_args.epochs = 0
    moefication_split_args.batch_size = 64
    moefication_split_args.model_args = {}
    moefication_split_args.model_args.num_experts = 128
    moefication_split_args.model_args.experts_class = 'execute_all'

    # ════════════════════════ moe split ════════════════════════ #

    base_moefication_exp_names = []
    for base_on_exp_name in base_relufication_exp_names:
        for exp_id in exp_ids:
            args = deepcopy(moefication_split_args)
            args.exp_id = exp_id
            args.base_on = base_on_exp_name
            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_on_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            # job = submit_job(executor, split_moe, args, num_gpus=moefication_gpus_per_task)
            # jobs.append(job)
            # run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        base_moefication_exp_names.append(exp_names[-1])
        display_names.append(f'MoEfication expert split')

    # ════════════════════════ router training model settings ════════════════════════ #

    moefication_gpus_per_task = 2
    executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}{moefication_gpus_per_task}')

    moefication_routing_args = deepcopy(common_args)
    moefication_routing_args.model_class = 'moefication_router'
    moefication_routing_args.router_loss_type = 'bcewl'
    moefication_routing_args.epochs = 10
    # moefication_routing_args.epochs = 0.01
    moefication_routing_args.batch_size = 256
    moefication_routing_args.optimizer_args.lr = 0.001
    moefication_routing_args.model_args = {}
    moefication_routing_args.model_args.depth = 2
    moefication_routing_args.model_args.width = 128
    moefication_routing_args.k_to_eval = [1, 2, 4, 8, 16, 32, 64, 128]
    # moefication_routing_args.k_to_eval = [64]
    moefication_routing_args.eval_points = 11
    # moefication_routing_args.eval_points = 0
    moefication_routing_args.mixed_precision = 'bf16'

    # ════════════════════════ moe router training ════════════════════════ #

    base_routed_moefication_exp_names = []
    for base_on_exp_name in base_moefication_exp_names:
        for exp_id in exp_ids:
            args = deepcopy(moefication_routing_args)
            args.exp_id = exp_id
            args.base_on = base_on_exp_name
            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_on_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            # job = submit_job(executor, train_routers, args, num_gpus=moefication_gpus_per_task)
            # jobs.append(job)
            # run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        base_routed_moefication_exp_names.append(exp_names[-1])
        display_names.append(f'MoEfication')

    # ════════════════════════ MHA replacement model settings ════════════════════════ #

    dsti_gpus_per_task = 2
    # dsti_gpus_per_task = 3
    # dsti_gpus_per_task = 4
    executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}{dsti_gpus_per_task}')

    distillation_args = deepcopy(common_args)
    distillation_args.model_class = 'mha_rep_distill'
    distillation_args.model_args = {}
    distillation_args.model_args.flops_factor = 1.0
    distillation_args.model_args.mlp_type = 'simple'
    distillation_args.model_args.dropout = 0.05
    distillation_args.base_on = base_exp_name
    distillation_args.epochs = 3
    # distillation_args.epochs = 0
    distillation_args.batch_size = 128
    # distillation_args.batch_size = 64
    distillation_args.dsti_distill_loss_type = 'mse'
    distillation_args.eval_points = 0
    distillation_args.optimizer_args.lr = 0.001
    #
    distillation_args.dsti_enforce_weight = 1e-3
    distillation_args.dsti_enforce_mode = 'relu_hoyer'
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
        # job = submit_job(executor, mha_distill, args, num_gpus=dsti_gpus_per_task)
        # jobs.append(job)
        # run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    base_distill_exp_names.append(exp_names[-1])
    display_names.append(f'MHA distillation')

    # ════════════════════════ sparsity enforcement settings ════════════════════════ #

    # dsti_gpus_per_task = 4
    # dsti_gpus_per_task = 3
    dsti_gpus_per_task = 2
    executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}{dsti_gpus_per_task}')

    sparsity_enforcement_args = deepcopy(common_args)
    sparsity_enforcement_args.dsti_enforce_mode = 'relu_hoyer'
    sparsity_enforcement_args.dsti_clamp_displacement = -10.0
    sparsity_enforcement_args.dsti_enforce_weight = 2e-1
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
    sparsity_enforcement_args.eval_points = 46
    # sparsity_enforcement_args.eval_points = 0
    sparsity_enforcement_args.optimizer_args.lr = 2e-5
    sparsity_enforcement_args.optimizer_args.weight_decay = 0.05
    # end LR for scheduler
    sparsity_enforcement_args.scheduler_args.eta_min = 1e-5
    sparsity_enforcement_args.mixed_precision = 'bf16'

    # ════════════════════════ activation sparsity enforcement ════════════════════════ #

    base_enforce_exp_names = []
    for base_on_exp_name in base_distill_exp_names:
    # for base_on_exp_name in [base_exp_name]:
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
            job = submit_job(executor, sparse_finetune, args, num_gpus=dsti_gpus_per_task)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        base_enforce_exp_names.append(exp_names[-1])
        display_names.append(f'Sparsity enforcement')

    # ════════════════════════ dsti moe split settings ════════════════════════ #

    # dsti_gpus_per_task = 4
    dsti_gpus_per_task = 2
    executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}{dsti_gpus_per_task}')

    expert_split_args = deepcopy(common_args)
    expert_split_args.model_class = 'dsti_expert_split'
    expert_split_args.epochs = 0
    expert_split_args.batch_size = 64
    expert_split_args.model_args = {}
    expert_split_args.model_args.expert_size = 6
    expert_split_args.model_args.experts_class = 'execute_all'

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
            job = submit_job(executor, dsti_expert_split, args, num_gpus=dsti_gpus_per_task)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        base_split_exp_names.append(exp_names[-1])
        display_names.append(f'DSTI expert split')

    # ════════════════════════ dsti router training model settings ════════════════════════ #

    # dsti_gpus_per_task = 4
    # dsti_gpus_per_task = 3
    dsti_gpus_per_task = 2
    executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}{dsti_gpus_per_task}')

    dsti_routing_args = deepcopy(common_args)
    dsti_routing_args.model_class = 'dsti_router'
    dsti_routing_args.router_loss_type = 'mse'
    dsti_routing_args.epochs = 7
    # dsti_routing_args.epochs = 0.1
    dsti_routing_args.batch_size = 128
    # dsti_routing_args.batch_size = 64
    dsti_routing_args.optimizer_args.lr = 0.001
    dsti_routing_args.model_args = {}
    dsti_routing_args.model_args.depth = 2
    dsti_routing_args.model_args.width = 128
    # dsti_routing_args.model_args.width = 32
    # dsti_routing_args.model_args.width = 16
    dsti_routing_args.model_args.activation = 'relu'
    dsti_routing_args.model_args.output_activation = 'abs'
    dsti_routing_args.router_labels_layer = 'output'
    dsti_routing_args.router_labels_norm = 2
    dsti_routing_args.dsti_tau_to_eval = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95,
                                          0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0]
    dsti_routing_args.dsti_expert_selection_mode = 'dynk_max'
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
            job = submit_job(executor, dsti_train_routers, args, num_gpus=dsti_gpus_per_task)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        base_routed_dsti_exp_names.append(exp_names[-1])
        display_names.append(f'DSTI')

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
        executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}1',
                                   slurm_additional_parameters={"dependency": dependency_str})
    else:
        executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}1', slurm_additional_parameters={})
    submit_job(executor, cost_vs_plot, plot_args)


if __name__ == '__main__':
    main()

