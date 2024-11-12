import argparse
import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.dynamic_sparsification.expert_split import train as dsti_expert_split
from methods.dynamic_sparsification.sparse_finetuning import train as sparse_finetune
from methods.dynamic_sparsification.train_routers import train as dsti_train_routers
from methods.moefication.expert_split import train as split_moe
from methods.moefication.relufication import train as relufication
from methods.moefication.train_routers import train as moefication_train_routers
from train import train
from utils import generate_run_name, submit_job
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot
from visualize.dsti_expert_stats import main as expert_stats
from visualize.dsti_expert_stats import get_default_args as get_default_expert_stats_args

N_GPUS = 1
SEQUENCE_LENGTH = 1024
EFFECTIVE_BATCH_SIZE = 64
BATCH_SIZE = 4
EVAL_BATCHES = 2
TEST_BATCH_SIZE = 4
TEST_BATCHES = 10
GRADIENT_ACCUMULATION_STEPS = int(EFFECTIVE_BATCH_SIZE / BATCH_SIZE)
FINETUNING_STEPS = 10
EVAL_POINTS = 0


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    n_tokens = FINETUNING_STEPS * EFFECTIVE_BATCH_SIZE * SEQUENCE_LENGTH
    n_tokens_b = n_tokens / 10 ** 9
    print(f"Training using {n_tokens_b}B tokens...")

    job_name = "gemma2b_d2d_relu"

    qos = "big"
    # account = "plgimpmoe-gpu-a100"
    # partition = "plgrid-gpu-a100"
    # partition = 'batch'
    partition = "dgxh100"
    account = None
    # partition = None

    # timeout = 60 * 24 * 7
    timeout = 60 * 24 * 4
    gpus_per_task = N_GPUS
    cpus_per_gpu = 4
    mem_per_gpu = "64G"
    gpu_type = ''

    executor = submitit.AutoExecutor(folder=os.environ["LOGS_DIR"])
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

    # ════════════════════════ common experiment settings ════════════════════════ #

    common_args = get_default_args()
    # exp_ids = [1, 2, 3]
    exp_ids = [1]

    # DATASET
    common_args.dataset = "c4"
    common_args.dataset_args = {
        "seq_length": SEQUENCE_LENGTH,
        "dataset_name": "allenai/c4",
        "tokenizer_name": "google/gemma-2b",
    }

    common_args.model_class = "gemma"
    common_args.model_args = {}
    common_args.model_args.model_name = "google/gemma-2b"

    # OPTIMIZATION
    common_args.last_batch = 0
    common_args.eval_points = 0
    common_args.batch_size = BATCH_SIZE
    common_args.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
    common_args.num_workers = 0
    common_args.mixed_precision = "bf16"
    common_args.eval_batches = EVAL_BATCHES
    common_args.test_batches = TEST_BATCHES
    common_args.test_batch_size = TEST_BATCH_SIZE
    common_args.save_every = 120

    common_args.loss_type = "ce"
    common_args.loss_args = {}
    common_args.loss_args.ignore_index = -1
    common_args.optimizer_class = "adam"
    common_args.optimizer_args = {}
    common_args.optimizer_args.lr = 1e-4  # ORG 6e-4
    common_args.optimizer_args.weight_decay = 0.01
    common_args.optimizer_args.betas = (0.9, 0.95)
    common_args.scheduler_class = "cosine_with_warmup"
    common_args.scheduler_args = {}
    common_args.scheduler_args.num_warmup_steps = FINETUNING_STEPS * 0.05
    common_args.scheduler_args.num_training_steps = FINETUNING_STEPS
    common_args.clip_grad_norm = 1.0
    common_args.mixed_precision = "bf16"

    # LOGGING
    common_args.runs_dir = Path(os.environ["RUNS_DIR"])
    common_args.use_wandb = True

    # ═══════════════════════════════════════════════════════════════════ #

    jobs = []
    run_to_job_map = {}
    exp_names = []
    display_names = []

    # ════════════════════════ base model settings ════════════════════════ #

    base_model_args = deepcopy(common_args)

    # ════════════════════════ evaluate base model ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(base_model_args)
        args.exp_id = exp_id
        exp_name, run_name = generate_run_name(args)
        job = submit_job(executor, train, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
        jobs.append(job)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append("Base Gemma evaluation")
    base_exp_name = exp_name

    # Relufication + sparsity enforcement
    dsti_relu_sparsification_args = deepcopy(base_model_args)
    dsti_relu_sparsification_args.model_class = "enforce_sparsity"
    dsti_relu_sparsification_args.last_batch = FINETUNING_STEPS
    dsti_relu_sparsification_args.eval_points = EVAL_POINTS
    dsti_relu_sparsification_args.base_on = base_exp_name
    dsti_relu_sparsification_args.model_args = {}
    dsti_relu_sparsification_args.dsti_enforce_mode = "relu_hoyer"
    dsti_relu_sparsification_args.dsti_enforce_weight = 0.0
    dsti_relu_sparsification_args.model_args.apply_to = "moe_eligible_only"

    enforce_args = [
        # {"dsti_enforce_weight": 0.0},
        # {"dsti_enforce_weight": 0.001},
        # {'dsti_enforce_weight': 0.005},
        # {"dsti_enforce_weight": 0.0001},
        {"dsti_enforce_weight": 0.0005},
    ]
    for run in enforce_args:
        dsti_enforce_weight = run["dsti_enforce_weight"]
        for exp_id in exp_ids:
            args = deepcopy(dsti_relu_sparsification_args)
            args.exp_id = exp_id
            args.dsti_enforce_weight = dsti_enforce_weight
            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, sparse_finetune, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            run_to_job_map[run_name] = job
        run["sparsification_exp_name"] = exp_name
        exp_names.append(exp_name)
        display_names.append((f"Gemma2B(ReLU), w={dsti_enforce_weight}"))

    executor.update_parameters(slurm_job_name="expert_split_" + job_name)

    dsti_relu_expert_split_args = deepcopy(dsti_relu_sparsification_args)
    dsti_relu_expert_split_args.model_class = "dsti_expert_split"
    dsti_relu_expert_split_args.epochs = 0
    dsti_relu_expert_split_args.model_args = {}
    dsti_relu_expert_split_args.model_args.experts_class = "execute_all"

    dsti_relu_base_arg = {
        "depth": 2,
        "width": 512,
        "batch_size": 16,
        "steps": 1000,
        "lr": 0.001,
        "activation": "relu",
        "output_activation": "abs",
        "target": "output",
        "norm": 2,
        "tau_to_eval": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
                        0.9, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.0],
    }
    # Max: 16384 neurons in intermediate layers
    dsti_relu_args = []
    for sparsification_args in enforce_args:
        enforce_base_exp_name = sparsification_args["sparsification_exp_name"]
        dsti_relu_args.extend(
            [
                # {
                #     **dsti_relu_base_arg,
                #     "num_experts": 8192,
                #     "base_exp_name": enforce_base_exp_name,
                #     "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                #     "batch_size": 2,
                #     "gradient_accumulation_steps": 8,
                #     "test_batch_size": TEST_BATCH_SIZE // 8,
                #     "eval_batches": EVAL_BATCHES * 8,
                #     "test_batches": TEST_BATCHES * 8,
                # },
                # {
                #     **dsti_relu_base_arg,
                #     "num_experts": 4096,
                #     "base_exp_name": enforce_base_exp_name,
                #     "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                #     "batch_size": 2,
                #     "gradient_accumulation_steps": 8,
                #     "eval_batches": EVAL_BATCHES * 8,
                #     "test_batch_size": TEST_BATCH_SIZE // 8,
                #     "test_batches": TEST_BATCHES * 8,
                # },
                # {
                #     **dsti_relu_base_arg,
                #     "num_experts": 2048,
                #     "base_exp_name": enforce_base_exp_name,
                #     "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                #     "batch_size": 4,
                #     "gradient_accumulation_steps": 4,
                #     "eval_batches": EVAL_BATCHES * 4,
                #     "test_batch_size": TEST_BATCH_SIZE // 4,
                #     "test_batches": TEST_BATCHES * 4,
                # },
                # {
                #     **dsti_relu_base_arg,
                #     "num_experts": 1024,
                #     "base_exp_name": enforce_base_exp_name,
                #     "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                #     "batch_size": 4,
                #     "gradient_accumulation_steps": 4,
                #     "eval_batches": EVAL_BATCHES * 4,
                #     "test_batch_size": TEST_BATCH_SIZE // 4,
                #     "test_batches": TEST_BATCHES * 4,
                # },
                # {
                #     **dsti_relu_base_arg,
                #     "num_experts": 512,
                #     "base_exp_name": enforce_base_exp_name,
                #     "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                #     "batch_size": 4,
                #     "gradient_accumulation_steps": 4,
                #     "eval_batches": EVAL_BATCHES * 4,
                #     "test_batch_size": TEST_BATCH_SIZE // 4,
                #     "test_batches": TEST_BATCHES * 4,
                # },
                # {
                #     **dsti_relu_base_arg,
                #     "num_experts": 256,
                #     "base_exp_name": enforce_base_exp_name,
                #     "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                #     "batch_size": 4,
                #     "gradient_accumulation_steps": 4,
                #     "eval_batches": EVAL_BATCHES * 4,
                #     "test_batch_size": TEST_BATCH_SIZE // 4,
                #     "test_batches": TEST_BATCHES * 4,
                # },
                # {
                #     **dsti_relu_base_arg,
                #     "num_experts": 128,
                #     "base_exp_name": enforce_base_exp_name,
                #     "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                #     "batch_size": 4,
                #     "gradient_accumulation_steps": 4,
                #     "eval_batches": EVAL_BATCHES * 4,
                #     "test_batch_size": TEST_BATCH_SIZE // 4,
                #     "test_batches": TEST_BATCHES * 4,
                # },
                {
                    **dsti_relu_base_arg,
                    "num_experts": 64,
                    "base_exp_name": enforce_base_exp_name,
                    "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                    "batch_size": 4,
                    "gradient_accumulation_steps": 4,
                    "eval_batches": EVAL_BATCHES * 4,
                    "test_batch_size": TEST_BATCH_SIZE // 4,
                    "test_batches": TEST_BATCHES * 4,
                },
            ]
        )

    for dsti_relu_arg in dsti_relu_args:
        dsti_enforce_weight = dsti_relu_arg["dsti_enforce_weight"]
        base_exp_name = dsti_relu_arg["base_exp_name"]
        num_experts = dsti_relu_arg["num_experts"]
        batch_size = dsti_relu_arg["batch_size"]
        eval_batches = dsti_relu_arg["eval_batches"]
        test_batches = dsti_relu_arg["test_batches"]
        test_batch_size = dsti_relu_arg["test_batch_size"]
        gradient_accumulation_steps = dsti_relu_arg["gradient_accumulation_steps"]
        for exp_id in exp_ids:
            args = deepcopy(dsti_relu_expert_split_args)
            args.exp_id = exp_id
            args.base_on = base_exp_name
            args.model_args.num_experts = num_experts
            args.batch_size = batch_size
            args.eval_batches = eval_batches
            args.test_batches = test_batches
            args.test_batch_size = test_batch_size
            args.gradient_accumulation_steps = gradient_accumulation_steps
            exp_name, run_name = generate_run_name(args)

            base_run_name = f"{base_exp_name}_{exp_id}"
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(
                    slurm_additional_parameters={"dependency": dependency_str}
                )
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, dsti_expert_split, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        dsti_relu_arg["expert_split_exp_name"] = exp_name
        display_names.append(
            f"DSTI(ReLU) split(ne={num_experts}, enf_weight={dsti_enforce_weight})"
        )

    executor.update_parameters(slurm_job_name="router_training_" + job_name)

    # ════════════════════════ DSTI(ReLU) router training model settings ════════════════════════ #
    dsti_relu_routers_args = deepcopy(dsti_relu_expert_split_args)
    dsti_relu_routers_args.model_class = "dsti_router"
    dsti_relu_routers_args.router_loss_type = "mse"
    dsti_relu_routers_args.dsti_expert_selection_mode = 'dynk_max'
    dsti_relu_routers_args.gradient_accumulation_steps = 1
    dsti_relu_routers_args.scheduler_args = {}
    dsti_relu_routers_args.scheduler_args.num_warmup_steps = 0
    dsti_relu_routers_args.model_args = {}
    dsti_relu_routers_args.eval_points = 0
    dsti_relu_routers_args.mixed_precision = "bf16"

    for dsti_relu_arg in dsti_relu_args:
        num_experts = dsti_relu_arg["num_experts"]
        dsti_relu_router_base_exp_name = dsti_relu_arg["expert_split_exp_name"]
        dsti_enforce_weight = dsti_relu_arg["dsti_enforce_weight"]
        router_depth = dsti_relu_arg["depth"]
        router_width = dsti_relu_arg["width"]
        batch_size = dsti_relu_arg["batch_size"]
        lr = dsti_relu_arg["lr"]
        last_batch = dsti_relu_arg["steps"]
        act = dsti_relu_arg["activation"]
        output_act = dsti_relu_arg["output_activation"]
        target = dsti_relu_arg["target"]
        norm = dsti_relu_arg["norm"]
        tau_to_eval = dsti_relu_arg["tau_to_eval"]

        gradient_accumulation_steps = dsti_relu_arg["gradient_accumulation_steps"]
        eval_batches = dsti_relu_arg["eval_batches"]
        test_batches = dsti_relu_arg["test_batches"]
        test_batch_size = dsti_relu_arg["test_batch_size"]

        for exp_id in exp_ids:
            args = deepcopy(dsti_relu_routers_args)
            args.ponder_loss_weight = 0.0
            args.exp_id = exp_id
            args.base_on = dsti_relu_router_base_exp_name
            args.model_args.depth = router_depth
            args.model_args.width = router_width
            args.batch_size = batch_size
            args.optimizer_args.lr = lr
            args.last_batch = last_batch
            args.gradient_accumulation_steps = gradient_accumulation_steps
            args.eval_batches = eval_batches
            args.test_batches = test_batches
            args.test_batch_size = test_batch_size

            args.model_args.activation = act
            args.model_args.output_activation = output_act
            args.dsti_router_labels_layer = target
            args.dsti_router_labels_norm = norm
            args.dsti_tau_to_eval = tau_to_eval

            exp_name, run_name = generate_run_name(args)
            base_run_name = f"{dsti_relu_router_base_exp_name}_{exp_id}"
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(
                    slurm_additional_parameters={"dependency": dependency_str}
                )
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, dsti_train_routers, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append(
            f"DSTI(ReLU) routers(enf={dsti_enforce_weight}, ne={num_experts}, steps={last_batch}, lr={lr}, {act.upper()}x{output_act.upper()}, t={target}, n={norm})"
        )

    print(f"Exp names: {exp_names}")
    print(f"Display names: {display_names}")
    print(f"SLURM JIDs: {[job.job_id for job in jobs]}")
    print(f"Run to JID mapping: {[(k, v.job_id) for k, v in run_to_job_map.items()]}")

    # ════════════════════════ plot cost vs acc ════════════════════════ #
    executor.update_parameters(slurm_job_name="plot_gemma_2b_d2d_relu_hoyer")

    out_dir_name = "gemma2b_d2d_relu_hoyer"
    output_dir = Path(os.environ["RESULTS_DIR"]) / out_dir_name

    plot_args = get_default_cost_plot_args()
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = display_names
    plot_args.output_name = "cost_vs"
    plot_args.mode = "acc"
    plot_args.use_wandb = False

    dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
    executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str},)
    submit_job(executor, cost_vs_plot, plot_args, gpu_type=gpu_type)

    plot_args = get_default_cost_plot_args()
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = display_names
    plot_args.output_name = "cost_vs"
    plot_args.mode = "loss"
    plot_args.use_wandb = False

    dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
    executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str},)
    submit_job(executor, cost_vs_plot, plot_args, gpu_type=gpu_type)

    expert_stats_args = get_default_expert_stats_args()
    expert_stats_args.output_dir = output_dir
    expert_stats_args.runs_dir = common_args.runs_dir
    expert_stats_args.exp_names = exp_names
    expert_stats_args.exp_ids = exp_ids
    expert_stats_args.display_names = display_names

    dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
    executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str},)
    submit_job(executor, expert_stats, expert_stats_args, gpu_type=gpu_type)


if __name__ == "__main__":
    main()
