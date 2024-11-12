import os
import pickle
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.dynamic_sparsification.expert_split import train as dsti_expert_split
from methods.dynamic_sparsification.train_routers import train as dsti_train_routers
from methods.dynamic_sparsification.sparse_finetuning import train as sparse_finetune
from methods.moefication.expert_split import train as moefication_expert_split
from methods.moefication.relufication import train as relufication
from methods.moefication.train_routers import train as moefication_train_routers
from train import train
from utils import generate_run_name, submit_job
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = "dsti_gpt2"

    # account = None
    qos = "normal"

    account = "plgimpmoe-gpu-a100"
    partition = "plgrid-gpu-a100"
    # partition = 'batch'
    account = None
    partition = None

    # timeout = 60 * 24 * 7
    timeout = 60 * 24 * 2
    gpus_per_task = 4
    cpus_per_gpu = 4
    mem_per_gpu = "64G"

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
    common_args.batch_size = 8
    common_args.num_workers = 0  # TODO check if this does not explode
    common_args.gradient_accumulation_steps = 120
    common_args.mixed_precision = "bf16"
    common_args.eval_points = 0
    common_args.eval_batches = 100  # TODO
    common_args.test_batches = 500  # TODO
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
    common_args.scheduler_args.num_warmup_steps = 2000
    common_args.clip_grad_norm = 1.0

    # LOGGING
    common_args.runs_dir = Path(os.environ["RUNS_DIR"])
    common_args.use_wandb = True

    meta_path = os.path.join(
        os.environ["GPT_DATA_DIR"], common_args.dataset, "meta.pkl"
    )
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
        job = submit_job(executor, train, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
        jobs.append(job)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append('Base GPT2')
    base_exp_name = exp_name
    # finetune base model
    ft_args = deepcopy(base_model_args)
    ft_args.last_batch = 1000
    ft_args.eval_points = 5
    ft_args.scheduler_class = "cosine_with_warmup"
    ft_args.scheduler_args = {}
    ft_args.scheduler_args.num_warmup_steps = 100
    for exp_id in exp_ids:
        args = deepcopy(ft_args)
        args.exp_id = exp_id
        exp_name, run_name = generate_run_name(args)
        job = submit_job(executor, train, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
        jobs.append(job)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append('Base GPT2 (finetuned)')

    # ════════════════════════ Moefication-RELU base model settings ════════════════════════ #
    moefication_relu_args = deepcopy(base_model_args)
    moefication_relu_args.model_class = "relufication"
    moefication_relu_args.last_batch = 1000
    moefication_relu_args.eval_points = 5
    moefication_relu_args.base_on = base_exp_name
    moefication_relu_args.scheduler_class = "cosine_with_warmup"
    moefication_relu_args.scheduler_args = {}
    moefication_relu_args.scheduler_args.num_warmup_steps = 100
    moefication_relu_args.model_args = {}
    moefication_relu_args.model_args.mode = "ffns_only"
    moefication_relu_names = []
    for exp_id in exp_ids:
        args = deepcopy(moefication_relu_args)
        args.exp_id = exp_id
        exp_name, run_name = generate_run_name(args)
        base_run_name = f'{base_exp_name}_{exp_id}'
        if base_run_name in run_to_job_map:
            dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
            executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
        else:
            executor.update_parameters(slurm_additional_parameters={})
        job = submit_job(executor, relufication, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
        jobs.append(job)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append('Relufication')
    moefication_relu_names.append(exp_names[-1])

    # ════════════════════════ DSTI-RELU base model ════════════════════════ #

    dsti_relu_sparsification_args = deepcopy(base_model_args)
    dsti_relu_sparsification_args.model_class = "enforce_sparsity"
    dsti_relu_sparsification_args.last_batch = 1000
    dsti_relu_sparsification_args.eval_points = 5
    dsti_relu_sparsification_args.base_on = base_exp_name
    dsti_relu_sparsification_args.scheduler_class = "cosine_with_warmup"
    dsti_relu_sparsification_args.scheduler_args = {}
    dsti_relu_sparsification_args.scheduler_args.num_warmup_steps = 100
    dsti_relu_sparsification_args.model_args = {}
    dsti_relu_sparsification_args.dsti_enforce_mode = 'relu_l1'
    dsti_relu_sparsification_args.dsti_enforce_weight = 5e-4
    dsti_relu_sparsification_args.model_args.apply_to = 'moe_eligible_only'
    dsti_relu_sparsification_names = []
    for exp_id in exp_ids:
        args = deepcopy(dsti_relu_sparsification_args)
        args.exp_id = exp_id
        args.base_on = base_exp_name
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
    exp_names.append(exp_name)
    display_names.append(('Sparsification (ReLU)'))
    dsti_relu_sparsification_names.append(exp_names[-1])

    # ════════════════════════ DSTI-GELU base model settings ════════════════════════ #

    dsti_gelu_sparsification_args = deepcopy(base_model_args)
    dsti_gelu_sparsification_args.model_class = "enforce_sparsity"
    dsti_gelu_sparsification_args.last_batch = 1000
    dsti_gelu_sparsification_args.eval_points = 5
    dsti_gelu_sparsification_args.base_on = base_exp_name
    dsti_gelu_sparsification_args.scheduler_args.num_warmup_steps = 100
    dsti_gelu_sparsification_args.model_args = {}
    dsti_gelu_sparsification_args.dsti_enforce_mode = 'gelu_preactivations_clamped'
    dsti_gelu_sparsification_args.dsti_clamp_displacement = -10
    dsti_gelu_sparsification_args.dsti_enforce_weight = 5e-5
    dsti_gelu_sparsification_args.model_args.apply_to = 'moe_eligible_only'
    dsti_gelu_sparsification_names = []
    for exp_id in exp_ids:
        args = deepcopy(dsti_gelu_sparsification_args)
        args.exp_id = exp_id
        args.base_on = base_exp_name
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
    exp_names.append(exp_name)
    display_names.append('Sparsification (GELU)')
    dsti_gelu_sparsification_names.append(exp_names[-1])
    # Use single GPU for MOEs
    gpus_per_task = 1

    # ════════════════════════ moefication expert split════════════════════════ #
    moefication_split_args = deepcopy(moefication_relu_args)
    moefication_split_args.model_class = 'moefication'
    moefication_split_args.epochs = 0
    moefication_split_args.model_args = {}
    moefication_split_args.model_args.experts_class = 'execute_all'
    moefication_split_args.test_batches = 500
    moefication_base_arg = {'depth': 2, 'width': 128, 'batch_size': 16, 'steps': 500, 'lr': 1e-3}
    moefication_args = [
        {**moefication_base_arg, 'num_experts': 128, 'batch_size': 16, 'k_to_eval': [8, 16, 32, 48, 64, 96, 128]}
    ]
    for moefication_arg in moefication_args:
        for base_exp_name in moefication_relu_names:
            num_experts = moefication_arg['num_experts']
            batch_size = moefication_arg['batch_size']
            for exp_id in exp_ids:
                args = deepcopy(moefication_split_args)
                args.exp_id = exp_id
                args.base_on = base_exp_name
                args.model_args.num_experts = num_experts
                args.batch_size = batch_size
                exp_name, run_name = generate_run_name(args)
                base_run_name = f'{base_exp_name}_{exp_id}'
                if base_run_name in run_to_job_map:
                    dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                    executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
                else:
                    executor.update_parameters(slurm_additional_parameters={})
                job = submit_job(executor, moefication_expert_split, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
                jobs.append(job)
                run_to_job_map[run_name] = job
            exp_names.append(exp_name)
            moefication_arg['base_exp_name'] = exp_name
            display_names.append(f'Moefication split(ne={num_experts})')

    # ════════════════════════ moe train════════════════════════ #
    moefication_routers_args = deepcopy(moefication_split_args)
    moefication_routers_args.model_class = 'moefication_router'
    moefication_routers_args.router_loss_type = 'bcewl'
    moefication_routers_args.last_batch = 500
    moefication_routers_args.gradient_accumulation_steps = 1
    moefication_routers_args.optimizer_args.lr = 0.001
    moefication_routers_args.scheduler_args = {}
    moefication_routers_args.scheduler_args.num_warmup_steps = 0
    moefication_routers_args.model_args = {}
    moefication_routers_args.model_args.activation = 'tanh'
    moefication_routers_args.model_args.output_activation = 'identity'
    moefication_routers_args.eval_points = 1
    moefication_routers_args.mixed_precision = 'bf16'

    for moefication_arg in moefication_args:
        moefication_base_exp_name = moefication_arg['base_exp_name']
        router_depth = moefication_arg['depth']
        router_width = moefication_arg['width']
        bs = moefication_arg["batch_size"]
        lr = moefication_arg['lr']
        last_batch = moefication_arg['steps']
        k_to_eval = moefication_arg['k_to_eval']
        for exp_id in exp_ids:
            args = deepcopy(moefication_routers_args)
            args.exp_id = exp_id
            args.base_on = moefication_base_exp_name
            args.model_args.depth = router_depth
            args.model_args.width = router_width
            args.batch_size = bs
            args.optimizer_args.lr = lr
            args.last_batch = last_batch
            args.k_to_eval = k_to_eval
            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, moefication_train_routers, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append(f'Moefication')

    # ════════════════════════ DSTI-RELU moe split════════════════════════ #
    dsti_relu_expert_split_args = deepcopy(dsti_relu_sparsification_args)
    dsti_relu_expert_split_args.model_class = 'dsti_expert_split'
    dsti_relu_expert_split_args.epochs = 0
    dsti_relu_expert_split_args.model_args = {}
    dsti_relu_expert_split_args.model_args.experts_class = 'execute_all'
    dsti_relu_expert_split_args.test_batches = 500

    dsti_relu_base_arg = {'depth': 2,
                          'width': 128,
                          'batch_size': 16,
                          'steps': 500,
                          'gradient_accumulation_steps': 1,
                          'lr': 1e-3,
                          'activation': 'relu',
                          'output_activation': 'abs',
                          'target': 'output',
                          'norm': 2,
                          'tau_to_eval': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
                          }
    dsti_relu_args = [
        {**dsti_relu_base_arg, 'num_experts': 768, 'batch_size': 8, 'lr': 0.003, 'steps': 5000},
    ]
    for dsti_relu_arg in dsti_relu_args:
        for base_exp_name in dsti_relu_sparsification_names:
            num_experts = dsti_relu_arg['num_experts']
            batch_size = dsti_relu_arg['batch_size']
            for exp_id in exp_ids:
                args = deepcopy(dsti_relu_expert_split_args)
                args.exp_id = exp_id
                args.base_on = base_exp_name
                args.model_args.num_experts = num_experts
                args.batch_size = batch_size
                exp_name, run_name = generate_run_name(args)
                base_run_name = f'{base_exp_name}_{exp_id}'
                if base_run_name in run_to_job_map:
                    dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                    executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
                else:
                    executor.update_parameters(slurm_additional_parameters={})
                job = submit_job(executor, dsti_expert_split, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
                jobs.append(job)
                run_to_job_map[run_name] = job
            exp_names.append(exp_name)
            dsti_relu_arg['base_exp_name'] = exp_name
            display_names.append(f'DSTI(RELU) split(ne={num_experts})')

    # ════════════════════════ DSTI(RELU) router training model settings ════════════════════════ #
    dsti_relu_routers_args = deepcopy(dsti_relu_expert_split_args)
    dsti_relu_routers_args.model_class = 'dsti_router'
    dsti_relu_routers_args.router_loss_type = 'mse'
    dsti_relu_routers_args.dsti_expert_selection_mode = 'dynk_max'
    dsti_relu_routers_args.last_batch = 500
    dsti_relu_routers_args.gradient_accumulation_steps = 1
    dsti_relu_routers_args.optimizer_args.lr = 0.001
    dsti_relu_routers_args.scheduler_args = {}
    dsti_relu_routers_args.scheduler_args.num_warmup_steps = 0
    dsti_relu_routers_args.model_args = {}
    dsti_relu_routers_args.eval_points = 1
    dsti_relu_routers_args.mixed_precision = 'bf16'

    for dsti_relu_arg in dsti_relu_args:
        dsti_relu_router_base_exp_name = dsti_relu_arg['base_exp_name']
        router_depth = dsti_relu_arg['depth']
        router_width = dsti_relu_arg['width']
        batch_size = dsti_relu_arg['batch_size']
        lr = dsti_relu_arg['lr']
        last_batch = dsti_relu_arg['steps']
        act = dsti_relu_arg['activation']
        output_act = dsti_relu_arg['output_activation']
        target = dsti_relu_arg['target']
        norm = dsti_relu_arg['norm']
        tau_to_eval = dsti_relu_arg['tau_to_eval']

        for exp_id in exp_ids:
            args = deepcopy(dsti_relu_routers_args)
            args.exp_id = exp_id
            args.base_on = dsti_relu_router_base_exp_name
            args.model_args.depth = router_depth
            args.model_args.width = router_width
            args.batch_size = batch_size
            args.optimizer_args.lr = lr
            args.last_batch = last_batch
            args.model_args.activation = act
            args.model_args.output_activation = output_act
            args.dsti_router_labels_layer = target
            args.dsti_router_labels_norm = norm
            args.dsti_tau_to_eval = tau_to_eval

            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, dsti_train_routers, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append(
            f'D2D(RELU)')

    # ════════════════════════ DSTI-GELU moe split════════════════════════ #
    dsti_gelu_expert_split_args = deepcopy(dsti_gelu_sparsification_args)
    dsti_gelu_expert_split_args.model_class = 'dsti_expert_split'
    dsti_gelu_expert_split_args.epochs = 0
    dsti_gelu_expert_split_args.model_args = {}
    dsti_gelu_expert_split_args.model_args.experts_class = 'execute_all'
    dsti_gelu_expert_split_args.test_batches = 500

    dsti_gelu_base_arg = {'depth': 2,
                          'width': 128,
                          'batch_size': 16,
                          'steps': 500,
                          'lr': 1e-3,
                          'gradient_accumulation_steps': 1,
                          'activation': 'relu',
                          'output_activation': 'abs',
                          'target': 'output',
                          'norm': 2,
                          'tau_to_eval': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
                          }
    dsti_gelu_args = [
        {**dsti_gelu_base_arg, 'num_experts': 768, 'batch_size': 8, 'lr': 0.005, 'steps': 2000},
    ]
    for dsti_gelu_arg in dsti_gelu_args:
        for base_exp_name in dsti_gelu_sparsification_names:
            num_experts = dsti_gelu_arg['num_experts']
            batch_size = dsti_gelu_arg['batch_size']
            for exp_id in exp_ids:
                args = deepcopy(dsti_gelu_expert_split_args)
                args.exp_id = exp_id
                args.base_on = base_exp_name
                args.model_args.num_experts = num_experts
                args.batch_size = batch_size
                exp_name, run_name = generate_run_name(args)

                base_run_name = f'{base_exp_name}_{exp_id}'
                if base_run_name in run_to_job_map:
                    dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                    executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
                else:
                    executor.update_parameters(slurm_additional_parameters={})
                job = submit_job(executor, dsti_expert_split, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
                jobs.append(job)
                run_to_job_map[run_name] = job
            exp_names.append(exp_name)
            dsti_gelu_arg['base_exp_name'] = exp_name
            display_names.append(f'D2D(GELU) split(ne={num_experts})')

    # ════════════════════════ DSTI(GELU) router training model settings ════════════════════════ #
    dsti_gelu_routers_args = deepcopy(dsti_gelu_expert_split_args)
    dsti_gelu_routers_args.model_class = 'dsti_router'
    dsti_gelu_routers_args.router_loss_type = 'mse'
    dsti_relu_routers_args.dsti_expert_selection_mode = 'dynk_max'
    dsti_gelu_routers_args.last_batch = 500
    dsti_gelu_routers_args.gradient_accumulation_steps = 1
    dsti_gelu_routers_args.optimizer_args.lr = 0.001
    dsti_gelu_routers_args.scheduler_args = {}
    dsti_gelu_routers_args.scheduler_args.num_warmup_steps = 0
    dsti_gelu_routers_args.model_args = {}
    dsti_gelu_routers_args.eval_points = 1
    dsti_gelu_routers_args.mixed_precision = 'bf16'

    for dsti_gelu_arg in dsti_gelu_args:
        dsti_gelu_router_base_exp_name = dsti_gelu_arg['base_exp_name']
        router_depth = dsti_gelu_arg['depth']
        router_width = dsti_gelu_arg['width']
        batch_size = dsti_gelu_arg['batch_size']
        lr = dsti_gelu_arg['lr']
        last_batch = dsti_gelu_arg['steps']
        act = dsti_gelu_arg['activation']
        output_act = dsti_gelu_arg['output_activation']
        target = dsti_gelu_arg['target']
        norm = dsti_gelu_arg['norm']
        tau_to_eval = dsti_gelu_arg['tau_to_eval']

        for exp_id in exp_ids:
            args = deepcopy(dsti_gelu_routers_args)
            args.exp_id = exp_id
            args.base_on = dsti_gelu_router_base_exp_name
            args.model_args.depth = router_depth
            args.model_args.width = router_width
            args.batch_size = batch_size
            args.optimizer_args.lr = lr
            args.last_batch = last_batch
            args.model_args.activation = act
            args.model_args.output_activation = output_act
            args.dsti_router_labels_layer = target
            args.dsti_router_labels_norm = norm
            args.dsti_tau_to_eval = tau_to_eval

            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, dsti_train_routers, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append(
            f'D2D(GELU)')

    print(f"Exp names: {exp_names}")
    print(f"Display names: {display_names}")
    print(f"SLURM JIDs: {[job.job_id for job in jobs]}")
    print(f"Run to JID mapping: {[(k, v.job_id) for k, v in run_to_job_map.items()]}")

    # ════════════════════════ plot cost vs acc ════════════════════════ #

    filtered_exp_names, filtered_display_names = [], []
    for exp_name, display_name in zip(exp_names, display_names):
        if 'split' not in display_name and 'Sparsification' not in display_name and 'Relufication' not in display_name and 'finetuned' not in display_name:
            filtered_exp_names.append(exp_name)
            filtered_display_names.append(display_name)

    print(f"Plotting for {filtered_display_names}")

    out_dir_name = f"gpt2_{common_args.dataset}_final_experiments"
    output_dir = Path(os.environ["RESULTS_DIR"]) / out_dir_name

    plot_args = get_default_cost_plot_args()
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = filtered_exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = filtered_display_names
    plot_args.output_name = "cost_vs"
    plot_args.mode = "acc"
    plot_args.use_wandb = False

    dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
    executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
    submit_job(executor, cost_vs_plot, plot_args, num_gpus=1)

    plot_args = get_default_cost_plot_args()
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = filtered_exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = filtered_display_names
    plot_args.output_name = "cost_vs"
    plot_args.mode = "loss"
    plot_args.use_wandb = False

    dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
    executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
    submit_job(executor, cost_vs_plot, plot_args, num_gpus=1)


if __name__ == "__main__":
    main()
