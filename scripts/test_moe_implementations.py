import logging
import os
from pathlib import Path

import submitit
import torch
import triton
from torch.profiler import profile, ProfilerActivity

from architectures.moe import moe_impls
from architectures.moe.moe_layers import CustomKernelExperts
from architectures.moe.moefication import MoeficationMoE, MoeficationRouter
from architectures.vit import MLP
from eval import benchmark_with_sample
from utils import submit_job, set_parameter_by_name


def check_if_standard_implementation_throws():
    batch_size = 1
    sequence_length = 197
    hidden_dim = 768
    expert_dim = 128
    num_experts = 24
    # test if it does not throw
    for device in ['cpu', 'cuda']:
        example_input = torch.randn(batch_size, sequence_length, hidden_dim, device=device)
        example_moe = MoeficationMoE('test_moe', hidden_dim, num_experts, True, expert_dim, torch.nn.GELU,
                                     experts_class='custom_kernel').to(device)
        example_gating_network = MoeficationRouter(hidden_dim, num_experts, width=128, depth=2, bias=False,
                                                   activation='relu', output_activation='abs').to(device)
        example_moe.router = example_gating_network
        example_moe.eval()
        example_moe(example_input)


def verify_implementation_correctness(mode):
    # batch_size = 1
    # batch_size = 256
    batch_size = 5
    # sequence_length = 197
    sequence_length = 2
    # hidden_dim = 1024
    # hidden_dim = 1
    # hidden_dim = 32
    hidden_dim = 64
    # expert_dim = 1024
    expert_dim = 32
    # expert_dim = 64
    # expert_dim = 128
    # expert_dim = 1
    # num_experts = 256
    # num_experts = 128
    # num_experts = 64
    # num_experts = 16
    num_experts = 4
    device = 'cuda'
    print(f'Verifying the "{mode}" implementation...')
    # torch.manual_seed(666)
    # TODO sometimes the first run gives high error
    # this is due to autotune not copying/resetting the tensors during tuning
    # see https://github.com/triton-lang/triton/issues/1563
    for i in range(3):
        example_input = torch.randn(batch_size, sequence_length, hidden_dim, device=device)
        example_moe = MoeficationMoE('test_moe', hidden_dim, num_experts, True, expert_dim, torch.nn.ReLU,
                                     experts_class='custom_kernel').to(device)
        example_gating_network = MoeficationRouter(hidden_dim, num_experts, width=128, depth=2, bias=False,
                                                   activation='relu', output_activation='abs').to(device)
        # example_gating_network = MoeficationRouter(hidden_dim, num_experts, width=128, depth=2, bias=False,
        #                                            activation='relu', output_activation='abs',
        #                                            test_override_outputs_p=0.5).to(device)
        example_moe.router = example_gating_network
        example_moe.forward_mode = 'dynk_max'
        example_moe.tau = 0.5
        example_moe.eval()
        example_gating_network.eval()
        correct_result, correct_gating_data = example_moe(example_input)
        example_moe.experts.forward_mode = mode
        # verify that the implementation does not introduce CPU-GPU synchronizations
        # which can negatively impact performance
        if mode in ['triton_atomic']:
            torch.cuda.set_sync_debug_mode("error")
        optimized_result, gating_data = example_moe(example_input)
        torch.cuda.set_sync_debug_mode("default")
        max_error = torch.max(torch.abs(correct_result - optimized_result))
        print(f'Maximum error of "{mode}" to "masked" implementation output: {max_error}')


def export_kernel_assembly():
    out_dir = Path.cwd() / f'kernel_assembly'
    out_dir.mkdir(exist_ok=True)
    # always list every triton kernel wrapped with autotune here
    for autotuner in [moe_impls.moe_first_kernel, moe_impls.moe_second_kernel_atomic,
                      moe_impls.moe_second_kernel, moe_impls.moe_merge_results_kernel,
                      moe_impls.moe_merge_results_batched_kernel]:
        kernel_name = autotuner.fn.__name__
        if not autotuner.fn.cache:
            print(f'Skipping {kernel_name} due to empty cache')
            continue
        # print(f'Cache: {autotuner.fn.cache}')
        out_file = out_dir / f'{kernel_name}.ptx'
        with out_file.open(mode='w') as f:
            print(list(autotuner.fn.cache[0].values())[0].asm['ptx'], file=f)
        print(f'Exported kernel assembly into {str(out_file)}.')
        # print the best config for each kernel
        if autotuner.best_config:
            print(f'Best config for {kernel_name}: {autotuner.best_config}')


def benchmark_implementations():
    # ViT-like dimensions
    sequence_length = 197
    hidden_dim = 768
    # hidden_dim = 1024
    # batch_sizes = [2 ** i for i in range(0, 11, 1)]
    # batch_sizes = [128]
    batch_sizes = [256]
    k_frac_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    for expert_dim, num_experts in [(128, 24)]:
        # for expert_dim, num_experts in [(128, 32)]:
        # for expert_dim, num_experts in [(128, 16), (128, 32),
        #                                 (64, 32), (64, 64),
        #                                 (32, 32), (32, 64), (32, 128),
        #                                 (16, 64), (16, 128), (16, 256),]:
        example_moe = MoeficationMoE('test_moe', hidden_dim, num_experts, True, expert_dim, torch.nn.ReLU,
                                     experts_class='custom_kernel').to('cuda')
        example_moe.eval()
        for k_frac in k_frac_list:
            mlp_hidden_dim = round(expert_dim * (num_experts * k_frac)) if k_frac != 0.0 else 1
            example_mlp = MLP(hidden_dim, [mlp_hidden_dim, hidden_dim]).to('cuda')
            example_mlp_compiled = torch.compile(example_mlp, fullgraph=True, dynamic=True, mode='max-autotune')
            example_mlp.eval()

            print(f'Benchmarking for {k_frac} fraction of average of executed experts'
                  f' out of {num_experts} total (expert_dim: {expert_dim}):')
            # TODO instantiate a router here that executes on average k_frac fraction of experts
            example_gating_network = MoeficationRouter(hidden_dim, num_experts, width=128, depth=2, bias=False,
                                                       activation='relu', output_activation='abs',
                                                       test_override_outputs_p=k_frac).to('cuda')
            example_gating_network.eval()
            example_moe.router = example_gating_network
            example_moe.forward_mode = 'dynk_max'
            example_moe.tau = 0.99
            flops_results = {}
            median_time_results = {}
            min_time_results = {}
            max_time_results = {}

            provider_keys = ['mlp', 'naive', 'triton_atomic']
            provider_display_names = ['MLP', 'Naive', 'Triton']

            # benchmark implementations
            @triton.testing.perf_report(
                triton.testing.Benchmark(
                    x_names=['batch_size'],  # Argument names to use as an x-axis for the plot.
                    x_vals=batch_sizes,  # Different possible values for `x_name`.
                    x_log=True,  # x axis is logarithmic.
                    line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
                    line_vals=provider_keys,
                    # Possible values for `line_arg`.
                    line_names=provider_display_names,
                    # Label name for the lines.
                    styles=[('blue', '-'), ('red', '-'), ('green', '-'), ('yellow', '-'), ('brown', '-')],
                    # Line styles.
                    ylabel='Runtime',  # Label name for the y-axis.
                    plot_name='moe-impl-performance',
                    # Name for the plot. Used also as a file name for saving the plot.
                    args={},  # Values for function arguments not in `x_names` and `y_name`.
                ))
            def benchmark(batch_size, provider):
                example_input = torch.randn(batch_size, sequence_length, hidden_dim, device='cuda')
                quantiles = [0.5, 0.025, 0.975]
                if provider == 'mlp':
                    with torch.inference_mode():
                        ms, min_ms, max_ms = triton.testing.do_bench(lambda: example_mlp_compiled(example_input),
                                                                     quantiles=quantiles, rep=2000)
                        estimated_flops = benchmark_with_sample(example_mlp, example_input)[0].total()
                else:
                    with torch.inference_mode():
                        example_moe.experts.forward_mode = 'naive'
                        estimated_flops = benchmark_with_sample(example_moe, example_input)[0].total()
                        example_moe.experts.forward_mode = provider
                        ms, min_ms, max_ms = triton.testing.do_bench(lambda: example_moe(example_input),
                                                                     quantiles=quantiles, rep=2000)
                secs = lambda msecs: msecs / 1000
                key = (batch_size, provider)
                flops_results[key] = estimated_flops / batch_size
                median_time_results[key] = secs(ms) / batch_size
                min_time_results[key] = secs(min_ms) / batch_size
                max_time_results[key] = secs(max_ms) / batch_size
                return ms / batch_size, max_ms / batch_size, min_ms / batch_size

            out_dir = Path.cwd() / 'benchmark_results' / f'moe_es_{expert_dim}_ne_{num_experts}' / f'{k_frac}_k_frac'
            out_dir.mkdir(exist_ok=True, parents=True)
            out_bench_path = out_dir / 'triton_bench'
            out_bench_path.mkdir(exist_ok=True)
            out_data_path = out_dir / 'results.pth'
            benchmark.run(print_data=True, save_path=str(out_bench_path))
            results = {'flops': flops_results, 'median_time': median_time_results,
                       'min_time': min_time_results, 'max_time': max_time_results,
                       'k_frac': k_frac, 'num_experts': num_experts}
            torch.save(results, out_data_path)
            print(f'Saved results into {str(out_data_path)}')


def weights_to(model, dtype):
    replacement_dict = {}
    for name, param in model.named_parameters():
        new_param = param.to(dtype)
        replacement_dict[name] = new_param
    for name, new_param in replacement_dict.items():
        set_parameter_by_name(model, name, new_param)


def benchmark_implementations_with_datatypes():
    # ViT-like dimensions
    sequence_length = 197
    hidden_dim = 768
    batch_sizes = [256]
    k_frac_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    # k_frac_list = [1.0]
    expert_dim = 128
    num_experts = 24
    for w_dtype, a_dtype in [
        (torch.float32, torch.float32),
        (torch.float16, torch.float16),
        # (torch.int8, torch.int8),
        # (torch.int8, torch.float32),
    ]:
        example_moe = example_moe = CustomKernelExperts(hidden_dim, num_experts, depth=2,
                                                        expert_dim=expert_dim, bias='without_last',
                                                        activation=torch.nn.ReLU).to('cuda')
        weights_to(example_moe, w_dtype)
        for k_frac in k_frac_list:
            print(f'Benchmarking for {k_frac} fraction of average of executed experts'
                  f' out of {num_experts} total (expert_dim: {expert_dim})'
                  f' for w_dtype {w_dtype} and a_dtype {a_dtype}:')

            median_time_results = {}
            min_time_results = {}
            max_time_results = {}

            # provider_keys = ['triton_atomic', 'triton']
            # provider_display_names = ['Triton (atomic)', 'Triton (merging)']
            provider_keys = ['triton_atomic']
            provider_display_names = ['Triton']

            # benchmark implementations
            @triton.testing.perf_report(
                triton.testing.Benchmark(
                    x_names=['batch_size'],  # Argument names to use as an x-axis for the plot.
                    x_vals=batch_sizes,  # Different possible values for `x_name`.
                    x_log=True,  # x axis is logarithmic.
                    line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
                    line_vals=provider_keys,
                    # Possible values for `line_arg`.
                    line_names=provider_display_names,
                    # Label name for the lines.
                    styles=[('blue', '-'), ('red', '-'), ('green', '-'), ('yellow', '-'), ('brown', '-')],
                    # Line styles.
                    ylabel='Runtime',  # Label name for the y-axis.
                    plot_name='moe-impl-performance',
                    # Name for the plot. Used also as a file name for saving the plot.
                    args={},  # Values for function arguments not in `x_names` and `y_name`.
                ))
            def benchmark(batch_size, provider):
                routing_tensor = torch.empty((batch_size, sequence_length, num_experts), device='cuda', dtype=a_dtype)
                routing_tensor.bernoulli_(k_frac)
                routing_tensor = routing_tensor.flatten(0, 1)
                if a_dtype in [torch.bfloat16, torch.float16, torch.float32, torch.float64]:
                    example_input = torch.randn(batch_size, sequence_length, hidden_dim, device='cuda', dtype=a_dtype)
                else:
                    example_input = torch.randint(-10, 10, (batch_size, sequence_length, hidden_dim),
                                                  device='cuda', dtype=a_dtype)
                example_input = example_input.flatten(0, 1)
                quantiles = [0.5, 0.025, 0.975]
                with torch.inference_mode():
                    example_moe.forward_mode = provider
                    ms, min_ms, max_ms = triton.testing.do_bench(lambda: example_moe(example_input, routing_tensor),
                                                                 quantiles=quantiles, rep=2000)
                secs = lambda msecs: msecs / 1000
                key = (batch_size, provider)
                median_time_results[key] = secs(ms) / batch_size
                min_time_results[key] = secs(min_ms) / batch_size
                max_time_results[key] = secs(max_ms) / batch_size
                return ms / batch_size, max_ms / batch_size, min_ms / batch_size

            out_dir = (Path.cwd() / 'benchmark_results' / f'moe_es_{expert_dim}_ne_{num_experts}' / f'{k_frac}_k_frac')
            out_dir.mkdir(exist_ok=True, parents=True)
            out_bench_path = out_dir / 'triton_bench'
            out_bench_path.mkdir(exist_ok=True)
            out_data_path = out_dir / 'results.pth'
            benchmark.run(print_data=True, save_path=str(out_bench_path))
            results = {'median_time': median_time_results, 'min_time': min_time_results, 'max_time': max_time_results,
                       'k_frac': k_frac, 'num_experts': num_experts}
            torch.save(results, out_data_path)
            print(f'Saved results into {str(out_data_path)}')

        # profile
        k_frac = 1.0

        def trace_handler(p):
            print(f'===== profiling results =====')
            print(p.key_averages().table(sort_by='self_cuda_time_total', row_limit=10))
            print(p.key_averages().table(sort_by='cpu_memory_usage', row_limit=10))
            out_dir = Path.cwd() / 'profiling_results'
            out_dir.mkdir(exist_ok=True)
            chrome_trace_path = out_dir / f'{a_dtype}_{w_dtype}_{p.step_num}.json'
            p.export_chrome_trace(str(chrome_trace_path))
            # profiler_stacks_path = out_dir / f'{mode}_{p.step_num}.txt'
            # p.export_stacks(str(profiler_stacks_path), 'self_cuda_time_total')

        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                schedule=torch.profiler.schedule(
                    skip_first=5,
                    wait=2,
                    warmup=2,
                    active=1),
                on_trace_ready=trace_handler
        ) as p:
            batch_size = 256
            num_batches = 10
            routing_tensor = torch.empty((batch_size, sequence_length, num_experts), device='cuda', dtype=a_dtype)
            routing_tensor.bernoulli_(k_frac)
            routing_tensor = routing_tensor.flatten(0, 1)
            if a_dtype in [torch.bfloat16, torch.float16, torch.float32, torch.float64]:
                example_input = torch.randn(batch_size, sequence_length, hidden_dim, device='cuda', dtype=a_dtype)
            else:
                example_input = torch.randint(-10, 10, (batch_size, sequence_length, hidden_dim),
                                              device='cuda', dtype=a_dtype)
            example_input = example_input.flatten(0, 1)
            for idx in range(num_batches):
                example_moe(example_input, routing_tensor)
                torch.cuda.synchronize(device=example_input.device)
                p.step()


def profile_implementation(mode):
    sequence_length = 197
    hidden_dim = 768
    # expert_dim = 32
    expert_dim = 128
    # num_experts = 64
    num_experts = 24
    batch_size = 256
    num_batches = 10

    # pre-generate batches
    example_inputs = [torch.randn(batch_size, sequence_length, hidden_dim,
                                  device='cuda') for _ in range(num_batches)]

    if mode == 'mlp':
        example_mlp = MLP(hidden_dim, [expert_dim * num_experts, hidden_dim]).to('cuda')
        example_mlp.eval()
    else:
        example_moe = MoeficationMoE('test_moe', hidden_dim, num_experts, True, expert_dim, torch.nn.ReLU,
                                     experts_class='custom_kernel').to('cuda')
        example_moe.eval()
        example_moe.experts.forward_mode = mode
        example_gating_network = MoeficationRouter(hidden_dim, num_experts, width=128, depth=2, bias=False,
                                                   activation='relu', output_activation='abs').to('cuda')
        example_moe.router = example_gating_network
        example_moe.forward_mode = 'dynk_max'
        example_moe.tau = 0.99
        example_moe.eval()

    def trace_handler(p):
        print(f'===== mode {mode} profiling results =====')
        print(p.key_averages().table(sort_by='self_cuda_time_total', row_limit=10))
        print(p.key_averages().table(sort_by='cpu_memory_usage', row_limit=10))
        out_dir = Path.cwd() / 'profiling_results'
        out_dir.mkdir(exist_ok=True)
        chrome_trace_path = out_dir / f'{mode}_{p.step_num}.json'
        p.export_chrome_trace(str(chrome_trace_path))
        # profiler_stacks_path = out_dir / f'{mode}_{p.step_num}.txt'
        # p.export_stacks(str(profiler_stacks_path), 'self_cuda_time_total')

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            schedule=torch.profiler.schedule(
                skip_first=5,
                wait=2,
                warmup=2,
                active=1),
            on_trace_ready=trace_handler
    ) as p:
        for idx in range(num_batches):
            if mode == 'mlp':
                example_mlp(example_inputs[idx])
            else:
                example_moe(example_inputs[idx])
            torch.cuda.synchronize(device=example_inputs[idx].device)
            p.step()


def run(_arg):
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info('Configured logging')
    #
    check_if_standard_implementation_throws()
    #
    for mode in ['naive', 'masked', 'triton_atomic', 'triton']:
        verify_implementation_correctness(mode)
    #
    export_kernel_assembly()
    #
    benchmark_implementations()
    benchmark_implementations_with_datatypes()
    export_kernel_assembly()
    #
    for mode in ['naive', 'masked', 'triton_atomic', 'triton']:
        profile_implementation(mode)


def main_slurm():
    job_name = 'effbench'

    # account = 'plgccbench-gpu-a100'
    # account = 'plgimpmoe-gpu-a100'
    account = None

    qos = 'normal'
    # qos = 'batch'

    # partition = 'plgrid-gpu-a100'
    # partition = 'dgxa100'
    # partition = 'dgxh100'
    # partition = 'rtx4090'
    # partition = 'rtx4090'
    partition = 'batch'

    timeout = 60 * 24 * 1
    # timeout = 60 * 24 * 2

    gpus_per_task = 1
    # gpu_type = ''
    gpu_type = 'ampere:'
    cpus_per_gpu = 8
    # cpus_per_gpu = 16
    # mem_per_gpu = '32G'
    # mem_per_gpu = '64G'
    mem_per_gpu = None

    exclude_nodes = None
    # exclude_nodes = 'gpu01'

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
    if exclude_nodes is not None:
        executor.update_parameters(slurm_exclude=exclude_nodes)

    job = submit_job(executor, run, None, num_gpus=1, gpu_type=gpu_type)
    print(f"SLURM JID: {job.job_id}")


if __name__ == '__main__':
    # useful for debugging
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['TORCH_CUDA_ALLOC_SYNC'] = '1'
    # os.environ['TRITON_DEBUG'] = '1'

    # deterministic matrix multiplication for CUDA >= 10.2
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # unfortunately, this throws for correctly working code
    # os.environ['TRITON_INTERPRET'] = '1'
    #
    # os.environ['TORCH_LOGS'] = 'perf_hints,dynamo,recompiles'
    # os.environ['TORCHDYNAMO_VERBOSE'] = '1'
    # torch._dynamo.config.cache_size_limit = 8
    main_slurm()
    # run(None)
