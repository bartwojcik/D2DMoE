import sys
if 'conda' in sys.path[-1] and 'site-packages' in sys.path[-1] and '.local' in sys.path[-2] and 'site-packages' in sys.path[-2]:
    print("Fixing sys.path")
    sys.path[-2], sys.path[-1] = sys.path[-1], sys.path[-2]

import os
import pickle
from copy import deepcopy
from pathlib import Path

import submitit
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args

from common import get_default_args
from methods.moefication.expert_split import train as split_moe
from methods.moefication.relufication import train as relufication
from methods.moefication.train_routers import train as train_routers
from train import train
from utils import generate_run_name, submit_job
from visualize import cost_vs_plot


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = "effbench_gpt2"

    account = None
    qos = "normal"
    partition = None

    # timeout = 60 * 24 * 7
    timeout = 60 * 24 * 2
    gpus_per_task = 8
    cpus_per_gpu = 4

    executor = submitit.AutoExecutor(folder=os.environ['LOGS_DIR'])
    executor.update_parameters(
        stderr_to_stdout=True,
        timeout_min=timeout,
        slurm_job_name=job_name,
        slurm_account=account,
        slurm_qos=qos,
        slurm_partition=partition,
        slurm_ntasks_per_node=1,
        slurm_gpus_per_task=gpus_per_task,
        slurm_cpus_per_gpu=cpus_per_gpu,
    )

    # ════════════════════════ common experiment settings ════════════════════════ #

    common_args = get_default_args()
    # exp_ids = [1, 2, 3]
    exp_ids = [1]

    block_size = 1024

    # DATASET
    common_args.dataset = "openwebtext"
    common_args.dataset_args = {"sequence_length": block_size}

    # MODEL
    common_args.model_class = "gpt2"
    common_args.model_args = {}
    common_args.model_args.model_name = "gpt2"
    common_args.model_args.block_size = block_size
    common_args.model_args.dropout = 0.1
    common_args.init_fun = None

    # OPTIMIZATION
    common_args.epochs = 0  # for finetuning from scratch, default is 60k steps
    common_args.batch_size = 16
    common_args.num_workers = 0 # TODO check if this does not explode
    common_args.gradient_accumulation_steps = 30
    common_args.mixed_precision = "bf16"
    common_args.eval_points = 0
    common_args.eval_batches = 100 # TODO
    common_args.test_batches = 200 # TODO
    common_args.save_every = 30  # save every 30 minuts since the model is fat

    common_args.loss_type = "ce"
    common_args.loss_args = {}
    common_args.loss_args.ignore_index = -1
    common_args.optimizer_class = "adam"
    common_args.optimizer_args = {}
    common_args.optimizer_args.lr = 6e-4
    common_args.optimizer_args.weight_decay = 1e-1
    common_args.optimizer_args.selective_weight_decay = True
    common_args.optimizer_args.betas = (0.9, 0.95)
    common_args.scheduler_class = "cosine_with_warmup"
    common_args.scheduler_args = {}
    common_args.scheduler_args.eta_min = 0.1 * common_args.optimizer_args.lr
    common_args.scheduler_args.warmup_epochs = 2000
    common_args.scheduler_args.warmup_start_lr = 0.0
    common_args.clip_grad_norm = 1.0

    # LOGGING
    common_args.runs_dir = Path(os.environ["RUNS_DIR"])
    common_args.use_wandb = True

    meta_path = os.path.join(
        os.environ["GPT_DATA_DIR"], common_args.dataset, "meta.pkl"
    )
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        common_args.model_args.meta_vocab_size = meta_vocab_size

    # ═══════════════════════════════════════════════════════════════════ #

    jobs = []
    run_to_job_map = {}
    exp_names = []
    display_names = []

    # ════════════════════════ base model settings ════════════════════════ #

    base_model_args = deepcopy(common_args)

    # ════════════════════════ train base models ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(base_model_args)
        args.exp_id = exp_id
        exp_name, run_name = generate_run_name(args)
        job = submit_job(executor, train, args, num_gpus=gpus_per_task)
        jobs.append(job)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append('GPT2')
    base_exp_name = exp_name

    # ════════════════════════ refulication model settings ════════════════════════ #

    relufication_args = deepcopy(base_model_args)
    relufication_args.last_batch = 3000
    relufication_args.scheduler_args.warmup_epochs = 200
    relufication_args.eval_points = 16
    relufication_args.model_class = "relufication"
    relufication_args.base_on = base_exp_name
    relufication_args.model_args = {}
    relufication_args.model_args.mode = "everywhere"

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
        job = submit_job(executor, relufication, args, num_gpus=gpus_per_task)
        jobs.append(job)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append('GPT2_relufication')
    base_relufication_exp_names.append(exp_names[-1])

    # ════════════════════════ moe split════════════════════════ #
    moefication_split_args = deepcopy(relufication_args)
    moefication_split_args.model_class = 'moefication'
    moefication_split_args.epochs = 0
    moefication_split_args.model_args = {}
    moefication_split_args.model_args.num_experts = 256
    moefication_split_args.model_args.experts_class = 'execute_all'

    base_moefication_exp_names = []
    for base_exp_name in base_relufication_exp_names:
        for exp_id in exp_ids:
            args = deepcopy(moefication_split_args)
            args.exp_id = exp_id
            args.base_on = base_exp_name
            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, split_moe, args, num_gpus=gpus_per_task)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append('GPT2_split_moe')
        base_moefication_exp_names.append(exp_names[-1])

    # ════════════════════════ moe train════════════════════════ #
    moefication_routing_args = deepcopy(moefication_split_args)
    moefication_routing_args.model_class = 'moefication_router'
    moefication_routing_args.router_loss_type = 'bcewl'
    moefication_routing_args.last_batch = 500
    moefication_routing_args.gradient_accumulation_steps = 1
    moefication_routing_args.batch_size = 32
    moefication_routing_args.optimizer_args.lr = 0.001
    moefication_routing_args.model_args = {}
    moefication_routing_args.model_args.depth = 2
    moefication_routing_args.model_args.width = 128
    moefication_routing_args.k_to_eval = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    moefication_routing_args.eval_points = 5

    base_routed_moefication_exp_names = []
    for base_exp_name in base_moefication_exp_names:
        for exp_id in exp_ids:
            args = deepcopy(moefication_routing_args)
            args.exp_id = exp_id
            args.base_on = base_exp_name
            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, train_routers, args, num_gpus=gpus_per_task)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append('GPT2_train_routers')
        base_routed_moefication_exp_names.append(exp_names[-1])

    print(f"Exp names: {exp_names}")
    print(f"Display names: {display_names}")
    print(f"SLURM JIDs: {[job.job_id for job in jobs]}")
    print(f"Run to JID mapping: {[(k, v.job_id) for k, v in run_to_job_map.items()]}")

    # ════════════════════════ plot cost vs acc ════════════════════════ #

    plot_args = get_default_cost_plot_args()

    out_dir_name = f"gpt2_{common_args.dataset}"
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
    executor.update_parameters(slurm_gpus_per_task=1,
                               slurm_additional_parameters={"dependency": dependency_str})
    submit_job(executor, cost_vs_plot, plot_args)


if __name__ == "__main__":
    main()
