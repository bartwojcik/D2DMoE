import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.avit import train as train_avit
from methods.dynamic_sparsification.expert_split import train as dsti_expert_split
from methods.dynamic_sparsification.rep_distill_train import train as mha_distill
from methods.dynamic_sparsification.sparse_finetuning import train as sparse_finetune
from methods.dynamic_sparsification.train_routers import train as dsti_train_routers
from methods.early_exit import train as train_ee
from methods.moefication.expert_split import train as split_moe
from methods.moefication.relufication import train as relufication
from methods.moefication.train_routers import train as train_routers
from train import train
from utils import generate_run_name, submit_job
from visualize.activation_sparsity import get_default_args as get_default_activation_sparsity_args
from visualize.activation_sparsity import main as activation_sparsity_plot
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot
from visualize.dsti_one_at_a_time import get_default_args as get_default_dsti_one_at_a_time_args
from visualize.dsti_one_at_a_time import main as dsti_one_at_a_time_plot
from visualize.moe_image_spatial_load import get_default_args as get_default_moe_image_spatial_load_args
from visualize.moe_image_spatial_load import main as dsti_spatial_load_plot


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
    # partition = 'dgx'
    # partition = 'rtx3080'
    # partition = 'batch'

    # timeout = 60 * 24 * 7
    timeout = 60 * 24 * 2

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
    common_args.epochs = 0.001
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
        job = submit_job(executor, train, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
        jobs.append(job)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append(f'ViT-B')
    base_exp_name = exp_name

    # ════════════════════════ ZTW model settings ════════════════════════ #

    ee_gpus_per_task = 2
    # ee_gpus_per_task = 4

    cascading_model_args = deepcopy(common_args)
    cascading_model_args.epochs = 0.001
    cascading_model_args.batch_size = 256
    cascading_model_args.with_backbone = False
    cascading_model_args.model_class = 'ztw_cascading'
    cascading_model_args.model_args = {}
    cascading_model_args.model_args.head_type = 'transformer_cascading_3l_head'
    cascading_model_args.model_args.place_at = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cascading_model_args.optimizer_args = {}
    cascading_model_args.optimizer_args.lr = 0.001
    cascading_model_args.optimizer_args.weight_decay = 0.00005
    cascading_model_args.mixed_precision = 'bf16'

    ensembling_model_args = deepcopy(common_args)
    ensembling_model_args.with_backbone = False
    ensembling_model_args.batch_size = 256
    ensembling_model_args.epochs = 0.001
    ensembling_model_args.model_class = 'ztw_ensembling'
    ensembling_model_args.model_args = {}
    ensembling_model_args.optimizer_class = 'adam'
    ensembling_model_args.optimizer_args = {}
    ensembling_model_args.optimizer_args.lr = 0.001
    ensembling_model_args.optimizer_args.weight_decay = 0.0

    # ════════════════════════ train ZTW model for comparison ════════════════════════ #

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
        job = submit_job(executor, train_ee, args, num_gpus=ee_gpus_per_task, gpu_type=gpu_type)
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
        job = submit_job(executor, train_ee, args, num_gpus=ee_gpus_per_task, gpu_type=gpu_type)
        jobs.append(job)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append(f'ZTW')

    # ════════════════════════ A-ViT model settings ════════════════════════ #

    avit_model_args = deepcopy(common_args)
    avit_model_args.epochs = 0.001
    avit_model_args.eval_points = 101
    # avit_model_args.eval_points = 0
    avit_model_args.model_class = 'avit'
    avit_model_args.model_args = {}
    avit_model_args.model_args.gate_scale = None
    avit_model_args.model_args.gate_center = None
    avit_model_args.ponder_loss_weight = None
    avit_model_args.distr_target_depth = None
    avit_model_args.distr_prior_loss_weight = None
    avit_model_args.optimizer_args = {}
    avit_model_args.optimizer_args.lr = None
    avit_model_args.optimizer_args.weight_decay = 0.0
    avit_model_args.clip_grad_norm = 1.5
    # in the "Experiments" section of the paper the authors state it is not compatible with mixup?
    avit_model_args.mixup_alpha = 0.0
    # what about cutmix?
    # authors also used AMP
    avit_model_args.mixed_precision = 'bf16'

    # ════════════════════════ train A-ViT model for comparison ════════════════════════ #

    # avit_gpus_per_task = 4
    avit_gpus_per_task = 2
    # avit_gpus_per_task = 1
    for lr, bs, distr_target_depth, gate_scale, gate_center, ponder_weight, distr_weight in [
        (0.001, 128, 4, 10.0, 75.0, 5e-4, 1.0),
    ]:
        for exp_id in exp_ids:
            args = deepcopy(avit_model_args)
            args.exp_id = exp_id
            args.optimizer_args.lr = lr
            args.batch_size = bs
            args.distr_target_depth = distr_target_depth
            args.model_args.gate_scale = gate_scale
            args.model_args.gate_center = gate_center
            args.ponder_loss_weight = ponder_weight
            args.distr_prior_loss_weight = distr_weight
            args.base_on = base_exp_name  # path to the base model will be inferred from the experiment name
            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, train_avit, args, num_gpus=avit_gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append(f'A-ViT (depth target {distr_target_depth})')

    # ════════════════════════ relufication model settings ════════════════════════ #

    moefication_gpus_per_task = 2

    relufication_args = deepcopy(common_args)
    relufication_args.model_class = 'relufication'
    relufication_args.model_args = {}
    relufication_args.model_args.mode = 'ffns_only'
    relufication_args.base_on = base_exp_name
    relufication_args.epochs = 0.001
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
        job = submit_job(executor, relufication, args, num_gpus=moefication_gpus_per_task, gpu_type=gpu_type)
        jobs.append(job)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    base_relufication_exp_names.append(exp_names[-1])
    display_names.append(f'ReLUfication')

    # ════════════════════════ moe split settings ════════════════════════ #

    moefication_gpus_per_task = 1

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
            job = submit_job(executor, split_moe, args, num_gpus=moefication_gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        base_moefication_exp_names.append(exp_names[-1])
        display_names.append(f'MoEfication expert split')

    # ════════════════════════ router training model settings ════════════════════════ #

    moefication_gpus_per_task = 2

    moefication_routing_args = deepcopy(common_args)
    moefication_routing_args.model_class = 'moefication_router'
    moefication_routing_args.router_loss_type = 'bcewl'
    moefication_routing_args.epochs = 0.001
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
            job = submit_job(executor, train_routers, args, num_gpus=moefication_gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        base_routed_moefication_exp_names.append(exp_names[-1])
        display_names.append(f'MoEfication')

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
    distillation_args.model_args.dropout = 0.05
    # distillation_args.model_args.dropout = 0.0
    distillation_args.base_on = base_exp_name
    distillation_args.epochs = 0.001
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
        job = submit_job(executor, mha_distill, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
        jobs.append(job)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    base_distill_exp_names.append(exp_names[-1])
    display_names.append(f'MHA distillation')

    # ════════════════════════ sparsity enforcement settings ════════════════════════ #

    # dsti_gpus_per_task = 4
    # dsti_gpus_per_task = 3
    dsti_gpus_per_task = 2

    sparsity_enforcement_args = deepcopy(common_args)
    sparsity_enforcement_args.dsti_enforce_mode = 'relu_hoyer'
    sparsity_enforcement_args.dsti_clamp_displacement = -10.0
    # sparsity_enforcement_args.dsti_enforce_weight = 5e-1
    sparsity_enforcement_args.dsti_enforce_weight = 2e-1
    # sparsity_enforcement_args.dsti_enforce_weight = 1e-1
    # sparsity_enforcement_args.dsti_enforce_weight = 5e-2
    # sparsity_enforcement_args.dsti_enforce_weight = 1e-2
    # sparsity_enforcement_args.dsti_enforce_weight = 5e-3
    # sparsity_enforcement_args.dsti_enforce_weight = 1e-3
    # sparsity_enforcement_args.dsti_enforce_weight = 2e-4
    #
    # sparsity_enforcement_args.dsti_enforce_schedule = 'linear'
    sparsity_enforcement_args.dsti_enforce_schedule = 'linear_warmup'
    sparsity_enforcement_args.model_class = 'enforce_sparsity'
    sparsity_enforcement_args.model_args = {}
    sparsity_enforcement_args.model_args.apply_to = 'moe_eligible_only'
    sparsity_enforcement_args.epochs = 0.001
    # sparsity_enforcement_args.epochs = 0.1
    sparsity_enforcement_args.batch_size = 128
    sparsity_enforcement_args.eval_points = 46
    # sparsity_enforcement_args.eval_points = 0
    sparsity_enforcement_args.optimizer_args.lr = 2e-5
    # sparsity_enforcement_args.optimizer_args.lr = 2e-4
    # sparsity_enforcement_args.optimizer_args.lr = 5e-4
    sparsity_enforcement_args.optimizer_args.weight_decay = 0.05
    # end LR for scheduler
    sparsity_enforcement_args.scheduler_args.eta_min = 1e-5
    # sparsity_enforcement_args.scheduler_args.eta_min = 1e-5
    # sparsity_enforcement_args.scheduler_args.eta_min = 2e-6
    #
    # sparsity_enforcement_args.mixed_precision = None
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
            job = submit_job(executor, sparse_finetune, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        base_enforce_exp_names.append(exp_names[-1])
        display_names.append(f'Sparsity enforcement')

    # ════════════════════════ dsti moe split settings ════════════════════════ #

    dsti_gpus_per_task = 1

    expert_split_args = deepcopy(common_args)
    expert_split_args.model_class = 'dsti_expert_split'
    expert_split_args.epochs = 0
    expert_split_args.batch_size = 64
    expert_split_args.model_args = {}
    # expert_split_args.model_args.expert_size = 32
    # expert_split_args.model_args.expert_size = 24
    # expert_split_args.model_args.expert_size = 12
    expert_split_args.model_args.expert_size = 6
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
        display_names.append(f'DSTI expert split')

    # ════════════════════════ dsti router training model settings ════════════════════════ #

    # dsti_gpus_per_task = 4
    dsti_gpus_per_task = 2

    dsti_routing_args = deepcopy(common_args)
    dsti_routing_args.model_class = 'dsti_router'
    dsti_routing_args.router_loss_type = 'mse'
    dsti_routing_args.epochs = 0.001
    dsti_routing_args.batch_size = 64
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
    dsti_routing_args.dsti_router_labels_norm = 2
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
            job = submit_job(executor, dsti_train_routers, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
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
        executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
    else:
        executor.update_parameters(slurm_additional_parameters={})
    submit_job(executor, cost_vs_plot, plot_args, num_gpus=1, gpu_type=gpu_type)

    # ════════════════════════ sparsity analysis ════════════════════════ #

    plot_args = get_default_activation_sparsity_args()
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = exp_names[5:6]
    plot_args.exp_ids = exp_ids
    plot_args.display_names = display_names[5:6]
    plot_args.use_wandb = False
    # plot_args.mixed_precision = None
    plot_args.mixed_precision = 'bf16'

    if jobs:
        dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
        executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
    else:
        executor.update_parameters(slurm_additional_parameters={})
    submit_job(executor, activation_sparsity_plot, plot_args, num_gpus=2, gpu_type=gpu_type)

    # ════════════════════════ dsti analysis ════════════════════════ #

    plot_args = get_default_dsti_one_at_a_time_args()
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = display_names
    plot_args.use_wandb = False
    # plot_args.mixed_precision = None
    plot_args.mixed_precision = 'bf16'
    if jobs:
        dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
        executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
    else:
        executor.update_parameters(slurm_additional_parameters={})
    submit_job(executor, dsti_one_at_a_time_plot, plot_args, num_gpus=2, gpu_type=gpu_type)

    # ════════════════════════ dsti analysis ════════════════════════ #

    plot_args = get_default_moe_image_spatial_load_args()
    plot_args.output_dir = output_dir / 'spatial_load'
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = display_names
    valid_set_len = 50_000
    images_to_draw = 80
    plot_args.data_indices = [round(i / images_to_draw * (valid_set_len - 1)) for i in range(images_to_draw)]
    plot_args.num_cheapest = 25
    plot_args.num_costliest = 25
    plot_args.batch_size = 32
    plot_args.dsti_tau = 0.9975
    plot_args.use_wandb = False
    if jobs:
        dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
        executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
    else:
        executor.update_parameters(slurm_additional_parameters={})
    submit_job(executor, dsti_spatial_load_plot, plot_args, num_gpus=1, gpu_type=gpu_type)


if __name__ == '__main__':
    main()
