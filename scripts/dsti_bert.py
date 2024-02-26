import sys


if 'conda' in sys.path[-1] and 'site-packages' in sys.path[-1] and '.local' in sys.path[-2] and 'site-packages' in \
        sys.path[-2]:
    print("Fixing sys.path")
    sys.path[-2], sys.path[-1] = sys.path[-1], sys.path[-2]

import os
import pickle
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.dynamic_sparsification.sparse_finetuning import train as sparse_finetune
from methods.dynamic_sparsification.expert_split import train as dsti_expert_split
from methods.dynamic_sparsification.train_routers import train as dsti_train_routers
from train import train
from utils import generate_run_name, submit_job
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot

DATASET_TO_SEQUENCE_LENGTH = {
    'rte': 128,
    'qqp': 128,
    'qnli': 128,
    'mrpc': 128,
    'sst2': 128,
    '20newsgroups': 512,
    'ag_news': 128,
    'dbpedia14': 128
}
DATASET_TO_NUM_CLASSES = {
    'rte': 2,
    'qqp': 2,
    'qnli': 2,
    'mrpc': 2,
    'sst2': 2,
    '20newsgroups': 20,
    'ag_news': 4,
    'dbpedia14': 14,
}

# change to any key from the dicts above to switch dataset
DATASET = "sst2"

MODEL_CLASS = 'bert'
TOKENIZER_NAME = 'bert-base-uncased'
DISPLAY_BASE = "BERT-base"
DS_ARGS = {'tokenizer_name': TOKENIZER_NAME, 'max_seq_length': DATASET_TO_SEQUENCE_LENGTH[DATASET]}
MODEL_ARGS = {'num_classes': DATASET_TO_NUM_CLASSES[DATASET], 'max_seq_length': DATASET_TO_SEQUENCE_LENGTH[DATASET]}
DS_TO_BASE_ARGS = {
    'rte': {'lr': 2e-5, 'batch_size': 16, 'epochs': 5, 'ee_batch_size': 32},
    'mrpc': {'lr': 2e-5, 'batch_size': 16, 'epochs': 5, 'ee_batch_size': 32},
    'sst2': {'lr': 1e-5, 'batch_size': 16, 'epochs': 3, 'ee_batch_size': 32},
    'qqp': {'lr': 5e-5, 'batch_size': 32, 'epochs': 3, 'ee_batch_size': 32},
    'qnli': {'lr': 1e-5, 'batch_size': 16, 'epochs': 3, 'ee_batch_size': 32},
    # TODO non-glue datasets hyperparams are not tested
    '20newsgroups': {'lr': 5e-5, 'batch_size': 32, 'epochs': 5, 'ee_batch_size': 64},
    'ag_news': {'lr': 5e-5, 'batch_size': 32, 'epochs': 5, 'ee_batch_size': 64},
    'dbpedia14': {'lr': 5e-5, 'batch_size': 32, 'epochs': 3, 'ee_batch_size': 64},
}
BASE_EPOCHS = DS_TO_BASE_ARGS[DATASET]['epochs']
BASE_LR = DS_TO_BASE_ARGS[DATASET]['lr']
BASE_BATCH_SIZE = DS_TO_BASE_ARGS[DATASET]['batch_size']
EE_BATCH_SIZE = DS_TO_BASE_ARGS[DATASET]['ee_batch_size']


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = "effbench_bert"

    account = None
    qos = "normal"
    partition = None

    # timeout = 60 * 24 * 7
    timeout = 60 * 24 * 2
    gpus_per_task = 1
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

    # LOGGING
    common_args.runs_dir = Path(os.environ["RUNS_DIR"])
    common_args.use_wandb = True

    # MODEL
    common_args.model_class = MODEL_CLASS
    common_args.TOKENIZER_NAME = MODEL_CLASS
    common_args.model_args = {}
    for k, v in MODEL_ARGS.items():
        setattr(common_args.model_args, k, v)
    common_args.model_args.model_name_or_path = 'bert-base-uncased'

    # DATASET
    common_args.dataset = DATASET
    common_args.dataset_args = DS_ARGS

    # OPTIMIZATION
    common_args.epochs = BASE_EPOCHS
    common_args.batch_size = BASE_BATCH_SIZE
    common_args.eval_points = 21
    common_args.eval_batches = 0
    common_args.loss_type = 'ce'
    common_args.loss_args = {}
    common_args.optimizer_class = 'adam'
    common_args.optimizer_args = {}
    common_args.optimizer_args.lr = BASE_LR
    common_args.optimizer_args.weight_decay = 0.0005
    common_args.scheduler_class = "cosine_with_warmup"
    common_args.scheduler_args = {}
    common_args.scheduler_args.eta_min = 0.1 * common_args.optimizer_args.lr
    common_args.scheduler_args.warmup_epochs = 50
    common_args.scheduler_args.warmup_start_lr = 0.0

    # ═══════════════════════════════════════════════════════════════════ #

    jobs = []
    run_to_job_map = {}
    exp_names = []
    display_names = []

    # ════════════════════════ base model settings ════════════════════════ #

    base_model_args = deepcopy(common_args)

    # ════════════════════════ finetune base model on downstream dataset ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(base_model_args)
        args.exp_id = exp_id
        job = submit_job(executor, train, args, num_gpus=gpus_per_task)
        jobs.append(job)
        exp_name, run_name = generate_run_name(args)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append('Bert')
    base_exp_name = exp_name

    # ════════════════════════ sparsification model settings ════════════════════════ #

    sparsification_args = deepcopy(base_model_args)
    sparsification_args.model_class = "enforce_sparsity"
    sparsification_args.base_on = base_exp_name
    sparsification_args.model_args = {}
    sparsification_args.dsti_enforce_mode = 'relu_hoyer'
    sparsification_args.dsti_enforce_weight = 1e-3
    sparsification_args.model_args.apply_to = 'moe_eligible_only'
    # sparsification_args.optimizer_args.lr = BASE_LR * 0.1 # Test later which sparsity fits well

    base_sparsification_names = []
    for exp_id in exp_ids:
        args = deepcopy(sparsification_args)
        args.exp_id = exp_id
        args.base_on = base_exp_name
        exp_name, run_name = generate_run_name(args)
        base_run_name = f'{base_exp_name}_{exp_id}'
        if base_run_name in run_to_job_map:
            dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
            executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
        else:
            executor.update_parameters(slurm_additional_parameters={})
        job = submit_job(executor, sparse_finetune, args, num_gpus=gpus_per_task)
        jobs.append(job)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append('Bert_sparsification')
    base_sparsification_names.append(exp_names[-1])

    # ════════════════════════ dsti moe split════════════════════════ #
    dsti_split_args = deepcopy(sparsification_args)
    dsti_split_args.model_class = 'dsti_expert_split'
    dsti_split_args.epochs = 0
    dsti_split_args.model_args = {}
    dsti_split_args.model_args.num_experts = 2
    dsti_split_args.model_args.experts_class = 'execute_all'

    base_dsti_split_names = []
    for base_exp_name in base_sparsification_names:
        for exp_id in exp_ids:
            args = deepcopy(dsti_split_args)
            args.exp_id = exp_id
            args.base_on = base_exp_name
            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, dsti_expert_split, args, num_gpus=gpus_per_task)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append('Bert_split_moe')
        base_dsti_split_names.append(exp_names[-1])

    # ════════════════════════ dsti router training model settings ════════════════════════ #
    dsti_routing_args = deepcopy(dsti_split_args)
    dsti_routing_args.model_class = 'dsti_router'
    dsti_routing_args.router_loss_type = 'mse'
    dsti_routing_args.epochs = 5
    dsti_routing_args.gradient_accumulation_steps = 1
    dsti_routing_args.batch_size = 64
    dsti_routing_args.optimizer_args.lr = 0.001
    dsti_routing_args.model_args = {}
    dsti_routing_args.model_args.depth = 2
    dsti_routing_args.model_args.width = 128
    dsti_routing_args.model_args.activation = 'gelu'
    dsti_routing_args.model_args.output_activation = 'abs'
    dsti_routing_args.dsti_tau_to_eval = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    dsti_routing_args.eval_points = 6

    base_routed_moefication_exp_names = []
    for base_exp_name in base_dsti_split_names:
        for exp_id in exp_ids:
            args = deepcopy(dsti_split_args)
            args.exp_id = exp_id
            args.base_on = base_exp_name
            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, dsti_train_routers, args, num_gpus=gpus_per_task)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append('Bert_train_routers')
        base_routed_moefication_exp_names.append(exp_names[-1])

    print(f"Exp names: {exp_names}")
    print(f"Display names: {display_names}")
    print(f"SLURM JIDs: {[job.job_id for job in jobs]}")
    print(f"Run to JID mapping: {[(k, v.job_id) for k, v in run_to_job_map.items()]}")

    # ════════════════════════ plot cost vs acc ════════════════════════ #

    plot_args = get_default_cost_plot_args()

    out_dir_name = f"bert_{common_args.dataset}"
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
