import argparse
import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.dynamic_sparsification.sparse_finetuning import train as sparse_finetune
from methods.moefication.expert_split import train as split_moe
from methods.moefication.train_routers import train as moefication_train_routers
from train import train
from utils import generate_run_name, submit_job
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot
from visualize.dsti_expert_stats import main as expert_stats
from visualize.dsti_expert_stats import get_default_args as get_default_expert_stats_args

N_GPUS = 1
SEQUENCE_LENGTH = 1024
EFFECTIVE_BATCH_SIZE = 1024
BATCH_SIZE = 4
EVAL_BATCHES = int(200000 / (SEQUENCE_LENGTH * BATCH_SIZE))
TEST_BATCH_SIZE = 16
TEST_BATCHES = int(2000000 / (SEQUENCE_LENGTH * TEST_BATCH_SIZE))
GRADIENT_ACCUMULATION_STEPS = int(EFFECTIVE_BATCH_SIZE / BATCH_SIZE)
FINETUNING_STEPS = 1000
EVAL_POINTS = 0


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    n_tokens = FINETUNING_STEPS * EFFECTIVE_BATCH_SIZE * SEQUENCE_LENGTH
    n_tokens_b = n_tokens / 10 ** 9
    print(f"Training using {n_tokens_b}B tokens...")

    job_name = "gemma2b_moefication_relu"

    # account = None
    qos = "big"
    mem_per_gpu = "64G"

    account = "plgimpmoe-gpu-a100"
    partition = "plgrid-gpu-a100"
    # partition = 'batch'
    partition = "dgxh100"
    account = None
    # partition = None

    # timeout = 60 * 24 * 7
    timeout = 60 * 24 * 4
    gpus_per_task = N_GPUS
    cpus_per_gpu = 4
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
    common_args.save_every = 120  # save every 10 minutes

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
        # job = submit_job(executor, train, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
        # jobs.append(job)
        # run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append("Base Gemma evaluation")
    base_exp_name = exp_name

    # Relufication + sparsity enforcement
    moefication_relu_sparsification_args = deepcopy(base_model_args)
    moefication_relu_sparsification_args.model_class = "enforce_sparsity"
    moefication_relu_sparsification_args.last_batch = FINETUNING_STEPS
    moefication_relu_sparsification_args.eval_points = EVAL_POINTS
    moefication_relu_sparsification_args.base_on = base_exp_name
    moefication_relu_sparsification_args.model_args = {}
    moefication_relu_sparsification_args.dsti_enforce_mode = "relu_hoyer"
    moefication_relu_sparsification_args.dsti_enforce_weight = 0.0
    moefication_relu_sparsification_args.model_args.apply_to = "moe_eligible_only"

    # moefication only runs for RELU model with no sparsification
    enforce_args = [
        {"dsti_enforce_weight": 0.0},
    ]
    for run in enforce_args:
        dsti_enforce_weight = run["dsti_enforce_weight"]
        for exp_id in exp_ids:
            args = deepcopy(moefication_relu_sparsification_args)
            args.exp_id = exp_id
            args.dsti_enforce_weight = dsti_enforce_weight
            exp_name, run_name = generate_run_name(args)
            # base_run_name = f'{base_exp_name}_{exp_id}'
            # if base_run_name in run_to_job_map:
            #     dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
            #     executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            # else:
            #     executor.update_parameters(slurm_additional_parameters={})
            # job = submit_job(executor, sparse_finetune, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
            # jobs.append(job)
            # run_to_job_map[run_name] = job
        run["sparsification_exp_name"] = exp_name
        exp_names.append(exp_name)
        display_names.append((f"Gemma2B(ReLU), w={dsti_enforce_weight}"))

    executor.update_parameters(slurm_job_name="expert_split_" + job_name)

    moefication_relu_expert_split_args = deepcopy(moefication_relu_sparsification_args)
    moefication_relu_expert_split_args.model_class = "moefication"
    moefication_relu_expert_split_args.epochs = 0
    moefication_relu_expert_split_args.model_args = {}
    moefication_relu_expert_split_args.model_args.experts_class = "execute_all"

    moefication_relu_base_arg = {
        "depth": 2,
        "width": 512,
        "batch_size": 16,
        "steps": 1000,
        "lr": 0.001,
        "activation": "relu",
        "output_activation": "abs",
    }
    moefication_relu_args = []
    for sparsification_args in enforce_args:
        enforce_base_exp_name = sparsification_args["sparsification_exp_name"]
        moefication_relu_args.extend(
            [
                {
                    **moefication_relu_base_arg,
                    "num_experts": 32,
                    "base_exp_name": enforce_base_exp_name,
                    "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                    "k_to_eval": [32, 28, 24, 20, 16, 12, 8, 4],
                    "batch_size": 16,
                    "gradient_accumulation_steps": 1,
                    'test_batch_size': TEST_BATCH_SIZE,
                    "eval_batches": EVAL_BATCHES,
                    "test_batches": TEST_BATCHES,
                },
                {
                    **moefication_relu_base_arg,
                    "num_experts": 64,
                    "base_exp_name": enforce_base_exp_name,
                    "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                    "k_to_eval": [64, 56, 48, 40, 32, 24, 16, 8],
                    "batch_size": 16,
                    "gradient_accumulation_steps": 1,
                    'test_batch_size': TEST_BATCH_SIZE,
                    "eval_batches": EVAL_BATCHES,
                    "test_batches": TEST_BATCHES,
                },
                {
                    **moefication_relu_base_arg,
                    "num_experts": 128,
                    "base_exp_name": enforce_base_exp_name,
                    "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                    "k_to_eval": [128, 112, 96, 80, 64, 48, 32, 16],
                    "batch_size": 16,
                    "gradient_accumulation_steps": 1,
                    'test_batch_size': TEST_BATCH_SIZE,
                    "eval_batches": EVAL_BATCHES,
                    "test_batches": TEST_BATCHES,
                },
                {
                    **moefication_relu_base_arg,
                    "num_experts": 256,
                    "base_exp_name": enforce_base_exp_name,
                    "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                    "k_to_eval": [256, 224, 192, 160, 128, 96, 64, 32],
                    "batch_size": 8,
                    "gradient_accumulation_steps": 2,
                    'test_batch_size': TEST_BATCH_SIZE // 2,
                    "eval_batches": EVAL_BATCHES * 2,
                    "test_batches": TEST_BATCHES * 2,
                },
                {
                    **moefication_relu_base_arg,
                    "num_experts": 512,
                    "base_exp_name": enforce_base_exp_name,
                    "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                    "k_to_eval": [512, 448, 384, 320, 256, 192, 128, 64],
                    "batch_size": 4,
                    "gradient_accumulation_steps": 4,
                    "eval_batches": EVAL_BATCHES * 4,
                    "test_batch_size": TEST_BATCH_SIZE // 4,
                    "test_batches": TEST_BATCHES * 4,
                },
                {
                    **moefication_relu_base_arg,
                    "num_experts": 1024,
                    "base_exp_name": enforce_base_exp_name,
                    "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                    "k_to_eval": [1024, 896, 768, 640, 512, 384, 256, 128],
                    "batch_size": 4,
                    "gradient_accumulation_steps": 4,
                    'test_batch_size': TEST_BATCH_SIZE // 4,
                    "eval_batches": EVAL_BATCHES * 4,
                    "test_batches": TEST_BATCHES * 4,
                },
                {
                    **moefication_relu_base_arg,
                    "num_experts": 2048,
                    "base_exp_name": enforce_base_exp_name,
                    "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                    "k_to_eval": [2048, 1792, 1536, 1280, 1024, 768, 512, 256],
                    "batch_size": 4,
                    "gradient_accumulation_steps": 4,
                    'test_batch_size': TEST_BATCH_SIZE // 4,
                    "eval_batches": EVAL_BATCHES * 4,
                    "test_batches": TEST_BATCHES * 4,
                },
                {
                    **moefication_relu_base_arg,
                    "num_experts": 4096,
                    "base_exp_name": enforce_base_exp_name,
                    "dsti_enforce_weight": sparsification_args["dsti_enforce_weight"],
                    "k_to_eval": [4096, 3584, 3072, 2560, 2048, 1536, 1024, 512],
                    "batch_size": 2,
                    "gradient_accumulation_steps": 8,
                    'test_batch_size': TEST_BATCH_SIZE // 8,
                    "eval_batches": EVAL_BATCHES * 8,
                    "test_batches": TEST_BATCHES * 8,
                },
            ]
        )

    for moefication_relu_arg in moefication_relu_args:
        dsti_enforce_weight = moefication_relu_arg["dsti_enforce_weight"]
        base_exp_name = moefication_relu_arg["base_exp_name"]
        num_experts = moefication_relu_arg["num_experts"]
        batch_size = moefication_relu_arg["batch_size"]
        eval_batches = moefication_relu_arg["eval_batches"]
        test_batches = moefication_relu_arg["test_batches"]
        test_batch_size = moefication_relu_arg["test_batch_size"]
        gradient_accumulation_steps = moefication_relu_arg[
            "gradient_accumulation_steps"
        ]
        for exp_id in exp_ids:
            args = deepcopy(moefication_relu_expert_split_args)
            args.exp_id = exp_id
            args.base_on = base_exp_name
            args.model_args.num_experts = num_experts
            args.batch_size = batch_size
            args.eval_batches = eval_batches
            args.test_batches = test_batches
            args.test_batch_size = test_batch_size
            args.gradient_accumulation_steps = gradient_accumulation_steps
            exp_name, run_name = generate_run_name(args)

            # base_run_name = f"{base_exp_name}_{exp_id}"
            # if base_run_name in run_to_job_map:
            #     dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
            #     executor.update_parameters(
            #         slurm_additional_parameters={"dependency": dependency_str}
            #     )
            # else:
            #     executor.update_parameters(slurm_additional_parameters={})
            # job = submit_job(executor, split_moe, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
            # jobs.append(job)
            # run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        moefication_relu_arg["expert_split_exp_name"] = exp_name
        display_names.append(
            f"Moefication(ReLU) split(ne={num_experts})"
        )

    # ════════════════════════ DSTI(ReLU) router training model settings ════════════════════════ #
    moefication_relu_routers_args = deepcopy(moefication_relu_expert_split_args)
    moefication_relu_routers_args.model_class = "moefication_router"
    moefication_relu_routers_args.router_loss_type = "bcewl"
    moefication_relu_routers_args.gradient_accumulation_steps = 1
    moefication_relu_routers_args.scheduler_args = {}
    moefication_relu_routers_args.scheduler_args.num_warmup_steps = 0
    moefication_relu_routers_args.model_args = {}
    moefication_relu_routers_args.eval_points = 0
    moefication_relu_routers_args.mixed_precision = "bf16"

    for moefication_relu_arg in moefication_relu_args:
        num_experts = moefication_relu_arg["num_experts"]
        dsti_relu_router_base_exp_name = moefication_relu_arg["expert_split_exp_name"]
        router_depth = moefication_relu_arg["depth"]
        router_width = moefication_relu_arg["width"]
        batch_size = moefication_relu_arg["batch_size"]
        lr = moefication_relu_arg["lr"]
        last_batch = moefication_relu_arg["steps"]
        act = moefication_relu_arg["activation"]
        output_act = moefication_relu_arg["output_activation"]
        k_to_eval = moefication_relu_arg["k_to_eval"]

        gradient_accumulation_steps = moefication_relu_arg[
            "gradient_accumulation_steps"
        ]
        eval_batches = moefication_relu_arg["eval_batches"]
        test_batches = moefication_relu_arg["test_batches"]
        test_batch_size = moefication_relu_arg["test_batch_size"]

        for exp_id in exp_ids:
            args = deepcopy(moefication_relu_routers_args)
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
            args.k_to_eval = k_to_eval

            exp_name, run_name = generate_run_name(args)
            base_run_name = f"{dsti_relu_router_base_exp_name}_{exp_id}"
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(
                    slurm_additional_parameters={"dependency": dependency_str}
                )
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(
                executor, moefication_train_routers, args, num_gpus=gpus_per_task
            )
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append(
            f"Moefication(ReLU) routers(ne={num_experts}, steps={last_batch}, lr={lr}, {act.upper()}x{output_act.upper()})"
        )

    print(f"Exp names: {exp_names}")
    print(f"Display names: {display_names}")
    print(f"SLURM JIDs: {[job.job_id for job in jobs]}")
    print(f"Run to JID mapping: {[(k, v.job_id) for k, v in run_to_job_map.items()]}")

    # ════════════════════════ plot cost vs acc ════════════════════════ #

    executor.update_parameters(slurm_job_name="plot_gemma_2b_moefication_relu")

    out_dir_name = "gemma2b_moefication_relu"
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
