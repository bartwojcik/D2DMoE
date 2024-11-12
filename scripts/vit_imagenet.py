import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
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

    qos = 'normal'
    # qos = 'big'

    # partition = 'plgrid-gpu-a100'
    # partition = 'dgxa100'
    # partition = 'rtx3080'
    partition = 'batch'

    timeout = 60 * 24 * 7
    # timeout = 60 * 24 * 2

    gpus_per_task = 1
    # gpu_type = ''
    gpu_type = 'ampere:'
    # cpus_per_gpu = 8
    cpus_per_gpu = 16
    mem_per_gpu = '64G'

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
    common_args.dataset_args.variant = 'trivial_augment'
    common_args.mixup_alpha = 0.8
    common_args.mixup_mode = 'elem'
    common_args.cutmix_alpha = 1.0
    common_args.mixup_smoothing = 0.1
    common_args.batch_size = None
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
    common_args.use_wandb = True

    jobs = []
    run_to_job_map = {}
    exp_names = []
    display_names = []

    # ════════════════════════ base model settings ════════════════════════ #
    # base_gpus_per_task = 4
    base_gpus_per_task = 6
    # base_gpus_per_task = 8

    base_model_args = deepcopy(common_args)
    base_model_args.model_class = 'vit'
    base_model_args.model_args = {}
    base_model_args.model_args.num_classes = 1000
    # ViT-B with patch size 16 on 224x224
    # base_model_args.model_args.image_size = 224
    # base_model_args.model_args.patch_size = 16
    # base_model_args.model_args.hidden_dim = 768
    # base_model_args.model_args.mlp_dim = 3072
    # base_model_args.model_args.num_layers = 12
    # base_model_args.model_args.num_heads = 12
    # ViT-S with patch size 16 on 224x224
    base_model_args.model_args.image_size = 224
    base_model_args.model_args.patch_size = 16
    base_model_args.model_args.hidden_dim = 384
    base_model_args.model_args.mlp_dim = 1536
    base_model_args.model_args.num_layers = 12
    base_model_args.model_args.num_heads = 6
    # ViT-T with patch size 16 on 224x224
    # base_model_args.model_args.image_size = 224
    # base_model_args.model_args.patch_size = 16
    # base_model_args.model_args.hidden_dim = 192
    # base_model_args.model_args.mlp_dim = 768
    # base_model_args.model_args.num_layers = 12
    # base_model_args.model_args.num_heads = 3

    # ════════════════════════ train base models ════════════════════════ #

    jobs = []
    exp_names = []
    display_names = []

    # quick hyperparam search
    # for lr, wd, clip_g_norm, a_dropout, dropout in [(0.001, 0.0, 1.0, 0.05, 0.05),
    #                                                 (0.0005, 0.0, 1.0, 0.05, 0.05),
    #                                                 (0.0001, 0.0, 1.0, 0.05, 0.05),
    #                                                 (0.00005, 0.0, 1.0, 0.05, 0.05),
    #                                                 (0.0005, 0.0, 1.0, 0.1, 0.1),
    #                                                 (0.0001, 0.0, 1.0, 0.1, 0.1),
    #                                                 (0.00005, 0.0, 1.0, 0.1, 0.1),
    #                                                 (0.0005, 0.0, 1.0, 0.0, 0.0),
    #                                                 (0.0001, 0.0, 1.0, 0.0, 0.0),
    #                                                 (0.00005, 0.0, 1.0, 0.0, 0.0),
    #                                                 ]:
    # best one for 1 GPU with 128 batch_size based on 5 epochs
    # for lr, wd, clip_g_norm, a_dropout, dropout in [(0.0001, 0.0, 1.0, 0.05, 0.05), ]:
    # multiple GPUs - square root scaling as proposed in https://arxiv.org/pdf/2006.09092.pdf
    # for lr, wd, clip_g_norm, a_dropout, dropout in [(0.0001 * (gpus_per_task ** 0.5), 0.0, 1.0, 0.05, 0.05), ]:
    # 75.07% acc:
    # for lr, wd, clip_g_norm, a_dropout, dropout in [(0.0005 * (gpus_per_task ** 0.5), 0.025, 1.0, 0.05, 0.05), ]:
    # TODO running acc:
    for epochs, bs, lr, wd, clip_g_norm, a_dropout, dropout, augment, precision in [
        # for ViT-S 4 GPUs
        # (500, 1024, 0.004, 0.05, 1.0, 0.05, 0.05, 'trivial_augment', 'bf16'),
        # for ViT-S 6 GPUs
        (1200, 1536, 0.004, 0.05, 1.0, 0.05, 0.05, 'trivial_augment', 'bf16'),
        # for ViT-S 8 GPUs
        # (1000, 2048, 0.004, 0.05, 1.0, 0.05, 0.05, 'trivial_augment', 'bf16'),

    ]:
        # for lr, wd, clip_g_norm, a_dropout, dropout in [(0.0001 * (gpus_per_task ** 0.5), 0.0, 1.0, 0.1, 0.1), ]:
        for exp_id in exp_ids:
            args = deepcopy(base_model_args)
            args.exp_id = exp_id
            args.epochs = epochs
            args.eval_points = epochs // 4 + 1
            args.batch_size = bs
            args.optimizer_args.lr = lr
            args.optimizer_args.weight_decay = wd
            args.clip_grad_norm = clip_g_norm
            args.model_args.attention_dropout = a_dropout
            args.model_args.dropout = dropout
            args.dataset_args.variant = augment
            args.mixed_precision = precision
            job = submit_job(executor, train, args, num_gpus=base_gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
    base_model_exp_name = generate_run_name(args)[0]
    exp_names.append(base_model_exp_name)
    display_names.append(f'ViT')

    # ═════════════════════════════════════════════════════════ #

    print(f"Exp names: {exp_names}")
    print(f"Display names: {display_names}")
    print(f"SLURM JIDs: {[job.job_id for job in jobs]}")

    # ════════════════════════ plot cost vs acc ════════════════════════ #

    plot_args = get_default_cost_plot_args()
    out_dir_name = f"vit_{common_args.dataset}"
    output_dir = Path(os.environ["RESULTS_DIR"]) / out_dir_name
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = display_names
    plot_args.output_name = "cost_vs"
    plot_args.mode = "acc"
    plot_args.use_wandb = False
    dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
    executor.update_parameters(stderr_to_stdout=True,
                               timeout_min=timeout,
                               slurm_job_name=job_name,
                               slurm_account=account,
                               slurm_qos=qos,
                               slurm_partition=partition,
                               slurm_ntasks_per_node=1,
                               slurm_cpus_per_gpu=cpus_per_gpu,
                               slurm_mem_per_gpu=mem_per_gpu,
                               slurm_additional_parameters={"dependency": dependency_str})
    submit_job(executor, cost_vs_plot, plot_args, gpu_type=gpu_type)


if __name__ == '__main__':
    main()
