import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.dynamic_sparsification.expert_split import train as dsti_expert_split
from methods.dynamic_sparsification.sparse_finetuning import train as sparse_finetune
from methods.dynamic_sparsification.train_routers import train as dsti_train_routers
from train import train
from utils import generate_run_name, submit_job
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = 'effbench'

    # account = 'plgccbench-gpu-a100'
    # account = 'plgimpmoe-gpu-a100'
    account = None

    # qos = 'normal'
    qos = 'big'

    # partition = 'plgrid-gpu-a100'
    # partition = 'dgxa100'
    partition = 'dgxh100'
    # partition = 'dgx'
    # partition = 'rtx3080'
    # partition = 'batch'

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
        job = submit_job(executor, train, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
        jobs.append(job)
        run_to_job_map[run_name] = job
    # exp_names.append(exp_name)
    # display_names.append(f'ViT-B')
    base_exp_name = exp_name

    # ════════════════════════ distill into a smaller model ════════════════════════ #

    distill_gpus_per_task = 4

    vit_s_model_args = deepcopy(common_args)
    vit_s_model_args.model_class = 'vit'
    vit_s_model_args.model_args = {}
    vit_s_model_args.model_args.image_size = 224
    vit_s_model_args.model_args.patch_size = 16
    vit_s_model_args.model_args.hidden_dim = 384
    vit_s_model_args.model_args.mlp_dim = 1536
    vit_s_model_args.model_args.num_layers = 12
    vit_s_model_args.model_args.num_heads = 6
    vit_s_model_args.model_args.num_classes = 1000

    vit_t_model_args = deepcopy(common_args)
    vit_t_model_args.model_class = 'vit'
    vit_t_model_args.model_args = {}
    vit_t_model_args.model_args.image_size = 224
    vit_t_model_args.model_args.patch_size = 16
    vit_t_model_args.model_args.hidden_dim = 192
    vit_t_model_args.model_args.mlp_dim = 768
    vit_t_model_args.model_args.num_layers = 12
    vit_t_model_args.model_args.num_heads = 3
    vit_t_model_args.model_args.num_classes = 1000

    for args, epochs, bs, lr, wd, clip_g_norm, a_dropout, dropout, augment, distill_weight, precision in [
        (vit_s_model_args, 400, 128, 0.004, 0.05, 1.0, 0.05, 0.05, 'trivial_augment', 1.0, 'bf16'),
        (vit_t_model_args, 400, 128, 0.004, 0.05, 1.0, 0.05, 0.05, 'trivial_augment', 1.0, 'bf16'),
    ]:
        for exp_id in exp_ids:
            args = deepcopy(args)
            args.exp_id = exp_id
            args.epochs = epochs
            args.eval_points = epochs // 8 + 1
            args.batch_size = bs
            args.optimizer_args.lr = lr
            args.optimizer_args.weight_decay = wd
            args.clip_grad_norm = clip_g_norm
            args.model_args.attention_dropout = a_dropout
            args.model_args.dropout = dropout
            args.dataset_args.variant = augment
            args.distill_from = base_exp_name
            args.distill_weight = distill_weight
            args.mixed_precision = precision
            exp_name, run_name = generate_run_name(args)
            job = submit_job(executor, train, args, num_gpus=distill_gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            run_to_job_map[run_name] = job
        distilled_exp_name = exp_name
        exp_names.append(exp_name)
        display_names.append(f'Distilled ViT')

        # ════════════════════════ sparsity enforcement settings ════════════════════════ #

        # dsti_gpus_per_task = 4
        # dsti_gpus_per_task = 3
        dsti_gpus_per_task = 2

        sparsity_enforcement_args = deepcopy(common_args)
        sparsity_enforcement_args.dsti_enforce_mode = 'relu_hoyer'
        sparsity_enforcement_args.dsti_clamp_displacement = -10.0
        # sparsity_enforcement_args.dsti_enforce_weight = 5e-1
        # sparsity_enforcement_args.dsti_enforce_weight = 2e-1
        # sparsity_enforcement_args.dsti_enforce_weight = 1e-1
        # sparsity_enforcement_args.dsti_enforce_weight = 5e-2
        # sparsity_enforcement_args.dsti_enforce_weight = 1e-2
        # sparsity_enforcement_args.dsti_enforce_weight = 5e-3
        # sparsity_enforcement_args.dsti_enforce_weight = 1e-3
        # 1st
        # sparsity_enforcement_args.dsti_enforce_weight = 2e-4
        # 2nd
        sparsity_enforcement_args.dsti_enforce_weight = 2e-5
        # sparsity_enforcement_args.dsti_enforce_weight = 1e-6
        #
        # sparsity_enforcement_args.dsti_enforce_schedule = 'linear'
        sparsity_enforcement_args.dsti_enforce_schedule = 'linear_warmup'
        # sparsity_enforcement_args.dsti_enforce_schedule = None
        sparsity_enforcement_args.model_class = 'enforce_sparsity'
        sparsity_enforcement_args.model_args = {}
        sparsity_enforcement_args.model_args.apply_to = 'moe_eligible_only'
        sparsity_enforcement_args.epochs = 90
        # sparsity_enforcement_args.epochs = 5
        # sparsity_enforcement_args.epochs = 0.1
        sparsity_enforcement_args.batch_size = 512
        # sparsity_enforcement_args.batch_size = 256
        # sparsity_enforcement_args.eval_points = 46
        sparsity_enforcement_args.eval_points = 0
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
        for base_on_exp_name in [distilled_exp_name]:
            for exp_id in exp_ids:
                args = deepcopy(sparsity_enforcement_args)
                args.exp_id = exp_id
                args.base_on = base_on_exp_name
                base_run_name = f'{base_on_exp_name}_{exp_id}'
                if base_run_name in run_to_job_map:
                    dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                    executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
                else:
                    executor.update_parameters(slurm_additional_parameters={})
                exp_name, run_name = generate_run_name(args)
                job = submit_job(executor, sparse_finetune, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
                jobs.append(job)
                run_to_job_map[run_name] = job
            exp_names.append(exp_name)
            base_enforce_exp_names.append(exp_names[-1])
            display_names.append(f'Sparsity enforcement')

        # ════════════════════════ dsti moe split settings ════════════════════════ #

        # dsti_gpus_per_task = 4
        dsti_gpus_per_task = 2

        expert_split_args = deepcopy(common_args)
        expert_split_args.model_class = 'dsti_expert_split'
        expert_split_args.epochs = 0
        expert_split_args.batch_size = 64
        expert_split_args.model_args = {}
        # expert_split_args.model_args.expert_size = 32
        # expert_split_args.model_args.expert_size = 24
        # expert_split_args.model_args.expert_size = 12
        expert_split_args.model_args.expert_size = 6
        expert_split_args.model_args.experts_class = 'execute_all'

        # ════════════════════════ dsti moe split ════════════════════════ #

        base_split_exp_names = []
        for base_on_exp_name in base_enforce_exp_names:
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

    out_dir_name = f"vit_{common_args.dataset}_wip_hoyer_distill"
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
    submit_job(executor, cost_vs_plot, plot_args, num_gpus=1, gpu_type=gpu_type)


if __name__ == '__main__':
    main()
