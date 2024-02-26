import sys

if (
    "conda" in sys.path[-1]
    and "site-packages" in sys.path[-1]
    and ".local" in sys.path[-2]
    and "site-packages" in sys.path[-2]
):
    print("Fixing sys.path")
    sys.path[-2], sys.path[-1] = sys.path[-1], sys.path[-2]

import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from datasets_config import DATASET_TO_NUM_CLASSES, DATASET_TO_SEQUENCE_LENGTH
from methods.dynamic_sparsification.expert_split import train as dsti_expert_split
from methods.dynamic_sparsification.sparse_finetuning import train as sparse_finetune
from methods.dynamic_sparsification.train_routers import train as dsti_train_routers
from methods.moefication.expert_split import train as moefication_expert_split
from methods.moefication.train_routers import train as moefication_train_routers
from train import train
from utils import generate_run_name, submit_job, summarize_experiments
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot

DATASET = "emotion"
MODEL_CLASS = "bert"
MODEL_CHECKPOINT = "relu-bert-base-uncased"
TOKENIZER_NAME = "bert-base-uncased"
NUM_CLASSES = DATASET_TO_NUM_CLASSES[DATASET]
MAX_SEQ_LEN = DATASET_TO_SEQUENCE_LENGTH[DATASET]

def main():
    # ════════════════════════ submitit setup ════════════════════════ #
    job_name = "effbench"
    account = None
    qos = "normal"
    partition = None
    timeout = 60 * 24 * 2
    gpus_per_task = 1
    cpus_per_gpu = 16
    mem_per_cpu = "6GB"

    executor = submitit.AutoExecutor(folder=os.environ["LOGS_DIR"])
    executor.update_parameters(
        stderr_to_stdout=True,
        slurm_time=timeout,
        slurm_job_name=job_name,
        slurm_account=account,
        slurm_qos=qos,
        slurm_partition=partition,
        slurm_ntasks_per_node=1,
        slurm_gpus_per_task=gpus_per_task,
        slurm_cpus_per_gpu=cpus_per_gpu,
        slurm_mem_per_cpu=mem_per_cpu,
    )

    # ════════════════════════ common experiment settings ════════════════════════ #

    common_args = get_default_args()
    exp_ids = [42]
    common_args.runs_dir = Path(os.environ["RUNS_DIR"])
    common_args.use_wandb = True
    common_args.model_class = MODEL_CLASS
    common_args.model_args = {"num_classes": NUM_CLASSES, "max_seq_length": MAX_SEQ_LEN}
    common_args.dataset = DATASET
    common_args.dataset_args = {"tokenizer_name": TOKENIZER_NAME, "max_seq_length": MAX_SEQ_LEN}
    common_args.loss_type = "ce"
    common_args.loss_args = {}
    common_args.loss_args.label_smoothing = 0.0
    common_args.optimizer_class = "adam"
    common_args.optimizer_args = {}
    common_args.gradient_accumulation_steps = 1
    common_args.scheduler_class = "linear"
    common_args.scheduler_args = {'num_warmup_steps': 0}
    common_args.mixed_precision = None
    common_args.save_every = 30
    common_args.eval_batches = 0
    common_args.batch_size = 32

    jobs = []
    run_to_job_map = {}
    exp_names = []
    display_names = []

    # ════════════════════════ finetuning on downstream dataset ════════════════════════ #

    finetuning_args = deepcopy(common_args)
    finetuning_args.model_args.model_name_or_path = MODEL_CHECKPOINT
    finetuning_args.optimizer_args.lr = 2e-5
    finetuning_args.optimizer_args.weight_decay = 0.0
    finetuning_args.gradient_accumulation_steps = 1
    finetuning_args.batch_size = 64
    finetuning_args.epochs = 5
    finetuning_args.eval_points = 20
    
    finetuning_exp_names = []
    for exp_id in exp_ids:
        args = deepcopy(finetuning_args)
        args.exp_id = exp_id
        exp_name, run_name = generate_run_name(args)
        job = submit_job(executor, train, args, num_gpus=gpus_per_task)
        jobs.append(job)
        run_to_job_map[run_name] = job

    exp_names.append(exp_name)
    display_names.append("BERT-base")
    finetuning_exp_names.append(exp_names[-1])

    # ════════════════════════ moefication expert construction ════════════════════════ #

    moefication_split_args = deepcopy(common_args)
    moefication_split_args.model_class = 'moefication'
    moefication_split_args.model_args = {}
    moefication_split_args.epochs = 0
    moefication_split_args.model_args.num_experts = 128
    moefication_split_args.model_args.experts_class = 'execute_all'

    moefication_expert_split_exp_names = []
    for base_on_exp_name in finetuning_exp_names:
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
            job = submit_job(executor, moefication_expert_split, args, num_gpus=gpus_per_task)
            jobs.append(job)
            run_to_job_map[run_name] = job

        exp_names.append(exp_name)
        moefication_expert_split_exp_names.append(exp_names[-1])
        display_names.append('MoEfication expert split')

    # ════════════════════════ moefication router training ════════════════════════ #

    moefication_routing_args = deepcopy(common_args)
    moefication_routing_args.model_class = 'moefication_router'
    moefication_routing_args.router_loss_type = 'bcewl'
    moefication_routing_args.epochs = 10
    moefication_routing_args.eval_points = 5
    moefication_routing_args.eval_batches = 30
    moefication_routing_args.batch_size = 64
    moefication_routing_args.optimizer_args.lr = 0.001
    moefication_routing_args.model_args = {}
    moefication_routing_args.model_args.depth = 2
    moefication_routing_args.model_args.width = 128
    moefication_routing_args.k_to_eval = [2, 32, 64, 96]
    moefication_routing_args.k_to_test = [1, 2, 4, 8, 16, 32, 64, 128]

    moefication_routing_exp_names = []
    for base_on_exp_name in moefication_expert_split_exp_names:
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
            job = submit_job(executor, moefication_train_routers, args, num_gpus=gpus_per_task)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        moefication_routing_exp_names.append(exp_names[-1])
        display_names.append('MoEfication')

    # ════════════════════════ get pretrained model checkpoint ════════════════════════ #

    # `enforce_sparsity` expects already pretrained model checkpoint to only modify its architecture
    # so we need to bypass it somehow
    base_model_args = deepcopy(common_args)
    base_model_args.model_args.model_name_or_path = MODEL_CHECKPOINT
    base_model_args.epochs = 0
    base_model_args.eval_points = 0

    for exp_id in exp_ids:
        args = deepcopy(base_model_args)
        args.exp_id = exp_id
        exp_name, run_name = generate_run_name(args)
        job = submit_job(executor, train, args, num_gpus=gpus_per_task)
        jobs.append(job)
        run_to_job_map[run_name] = job

    base_exp_name = exp_name

    # ════════════════════════ sparse finetuning ════════════════════════ #

    sparsification_args = deepcopy(finetuning_args)
    sparsification_args.model_class = "enforce_sparsity"
    sparsification_args.model_args = {}
    sparsification_args.model_args.apply_to = "moe_eligible_only"
    sparsification_args.dsti_enforce_mode = "relu_l1"
    sparsification_args.dsti_enforce_weight = 1e-4

    sparsification_exp_names = []
    for exp_id in exp_ids:
        args = deepcopy(sparsification_args)
        args.exp_id = exp_id
        args.base_on = base_exp_name
        exp_name, run_name = generate_run_name(args)
        base_run_name = f"{base_exp_name}_{exp_id}"
        if base_run_name in run_to_job_map:
            dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
            executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
        else:
            executor.update_parameters(slurm_additional_parameters={})
        job = submit_job(executor, sparse_finetune, args, num_gpus=gpus_per_task)
        jobs.append(job)
        run_to_job_map[run_name] = job

    exp_names.append(exp_name)
    display_names.append("Sparsity enforcement")
    sparsification_exp_names.append(exp_names[-1])

    # ════════════════════════ dsti expert construction ════════════════════════ #

    dsti_split_args = deepcopy(common_args)
    dsti_split_args.model_class = "dsti_expert_split"
    dsti_split_args.model_args = {}
    dsti_split_args.model_args.num_experts = 128
    dsti_split_args.model_args.experts_class = "execute_all"
    dsti_split_args.epochs = 0
    dsti_split_args.eval_points = 0

    dsti_split_exp_names = []
    for base_exp_name in sparsification_exp_names:
        for exp_id in exp_ids:
            args = deepcopy(dsti_split_args)
            args.exp_id = exp_id
            args.base_on = base_exp_name
            exp_name, run_name = generate_run_name(args)
            base_run_name = f"{base_exp_name}_{exp_id}"
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, dsti_expert_split, args, num_gpus=gpus_per_task)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append("DSTI expert split")
        dsti_split_exp_names.append(exp_names[-1])

    # ════════════════════════ dsti router training model settings ════════════════════════ #

    dsti_routing_args = deepcopy(common_args)
    dsti_routing_args.model_class = "dsti_router"
    dsti_routing_args.model_args = {}
    dsti_routing_args.model_args.depth = 2
    dsti_routing_args.model_args.width = 128
    dsti_routing_args.model_args.activation = "relu"
    dsti_routing_args.model_args.output_activation = "abs"
    dsti_routing_args.router_loss_type = "mse"
    dsti_routing_args.router_labels_norm = 1
    dsti_routing_args.router_labels_layer = "intermediate"
    dsti_routing_args.epochs = 10
    dsti_routing_args.eval_points = 5
    dsti_routing_args.eval_batches = 30
    dsti_routing_args.dsti_tau_to_eval = [0.25, 0.5, 0.75, 0.95]
    dsti_routing_args.dsti_tau_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    dsti_routing_args.batch_size = 64
    dsti_routing_args.optimizer_args.lr = 0.001

    dsti_routing_exp_names = []
    for base_exp_name in dsti_split_exp_names:
        for exp_id in exp_ids:
            args = deepcopy(dsti_routing_args)
            args.exp_id = exp_id
            args.base_on = base_exp_name
            exp_name, run_name = generate_run_name(args)
            base_run_name = f"{base_exp_name}_{exp_id}"
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, dsti_train_routers, args, num_gpus=gpus_per_task)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append("DSTI")
        dsti_routing_exp_names.append(exp_names[-1])

    # ═════════════════════════════════════════════════════════ #

    run_to_job_id = {k: v.job_id for k, v in run_to_job_map.items()}
    summarize_experiments(exp_names, display_names, run_to_job_id)

    # ════════════════════════ plot cost vs acc ════════════════════════ #

    plot_args = get_default_cost_plot_args()

    out_dir_name = f"bert_moefication_dsti_{common_args.dataset}"
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
    executor.update_parameters(slurm_gpus_per_task=1, slurm_additional_parameters={"dependency": dependency_str})
    submit_job(executor, cost_vs_plot, plot_args)


if __name__ == "__main__":
    main()
