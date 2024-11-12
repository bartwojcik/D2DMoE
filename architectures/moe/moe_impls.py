import torch
import triton
from triton import language as tl

from utils import row_major, column_major, grouped, leaky_relu, relu, tanh, gelu, config_grid


class MoeFirstLayerImplementation(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, bias, sort_indices, expert_bincounts, activation_str):
        sample_dim = input.size(0)
        num_experts = weight.size(0)
        hidden_dim = weight.size(-2)
        expert_dim = weight.size(-1)
        out = torch.empty((num_experts, sample_dim, expert_dim), device=input.device, dtype=input.dtype)
        grid = lambda META: (triton.cdiv(sample_dim, META['BLOCK_SIZE_BD']) *
                             triton.cdiv(expert_dim, META['BLOCK_SIZE_ED']),
                             num_experts)
        # commented out fragments used for the
        # assert weight.stride(2) == 1, f'{weight.stride(2)=}'
        # assert weight.stride(1) > 1, f'{weight.stride(1)=}'
        # if input.dtype == torch.int8:
        #     # tensor cores require column-major ordering for optimal performance
        #     # for B/weight matrix when using INT8 etc.
        #     # so we temporarily switch the strides for benchmarking purposes
        #     moe_first_kernel[grid](input, input.stride(0), input.stride(1),
        #                            weight, weight.stride(0), weight.stride(2), weight.stride(1),
        #                            bias, bias.stride(0), bias.stride(1),
        #                            out, out.stride(0), out.stride(1), out.stride(2),
        #                            sort_indices, sort_indices.stride(0), sort_indices.stride(1),
        #                            expert_bincounts,
        #                            sample_dim,
        #                            hidden_dim,
        #                            expert_dim,
        #                            NUM_EXPERTS=num_experts,
        #                            ACTIVATION=activation_str,
        #                            )
        # else:
        moe_first_kernel[grid](input, input.stride(0), input.stride(1),
                               weight, weight.stride(0), weight.stride(1), weight.stride(2),
                               bias, bias.stride(0), bias.stride(1),
                               out, out.stride(0), out.stride(1), out.stride(2),
                               sort_indices, sort_indices.stride(0), sort_indices.stride(1),
                               expert_bincounts,
                               sample_dim,
                               hidden_dim,
                               expert_dim,
                               NUM_EXPERTS=num_experts,
                               ACTIVATION=activation_str,
                               )
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        (input, weight, bias, sort_indices, expert_bincounts, activation_str) = inputs
        ctx.activation_str = activation_str
        ctx.save_for_backward(input, weight, bias, sort_indices, expert_bincounts)

    @staticmethod
    def backward(ctx, output_grad):
        (input, weight, bias, sort_indices, expert_bincounts) = ctx.saved_tensors
        activation_str = ctx.activation_str
        raise NotImplementedError('TODO')


@triton.autotune(
    configs=config_grid({
        'BLOCK_SIZE_BD': [32, 64, 128],
        'BLOCK_SIZE_HD': [64, 128, 256],
        'BLOCK_SIZE_ED': [32, 64],
        'GROUP_SIZE_BD': [4, 8, 16],
        'ORDERING': ['GROUPED']},
        num_stages=3, num_warps=8) +
            config_grid({
                'BLOCK_SIZE_BD': [32, 64, 128],
                'BLOCK_SIZE_HD': [32, 64, 128, 256],
                'BLOCK_SIZE_ED': [32, 64],
                'GROUP_SIZE_BD': [1],
                'ORDERING': ['ROW_MAJOR']},
                num_stages=4, num_warps=4) +
            config_grid({
                'BLOCK_SIZE_BD': [32, 64, 128],
                'BLOCK_SIZE_HD': [32, 64, 128, 256],
                'BLOCK_SIZE_ED': [32, 64],
                'GROUP_SIZE_BD': [1],
                'ORDERING': ['COLUMN_MAJOR']},
                num_stages=4, num_warps=4),
    # configs=[triton.Config({'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 64, 'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1,
    #                         'ORDERING': 'ROW_MAJOR'}, num_stages=4, num_warps=4)],
    key=['sample_dim', 'hidden_dim', 'expert_dim', 'NUM_EXPERTS'],
)
@triton.jit
def moe_first_kernel(x_ptr, stride_x_bd, stride_x_hd,
                     # bd - batch dim, hd - hidden dim
                     # ned - num experts dim, ed - expert dim
                     # tepd - token-expert pair dimension
                     weight_ptr, stride_weight_ned, stride_weight_hd, stride_weight_ed,
                     bias_ptr, stride_bias_ned, stride_bias_ed,
                     output_ptr, stride_output_ned, stride_output_bd, stride_output_ed,
                     # metadata
                     sort_indices_ptr, stride_sort_indices_bd, stride_sort_indices_ned,
                     expert_bincounts_ptr,
                     sample_dim,
                     hidden_dim,
                     expert_dim,
                     NUM_EXPERTS: tl.constexpr,
                     ACTIVATION: tl.constexpr,
                     BLOCK_SIZE_BD: tl.constexpr, BLOCK_SIZE_HD: tl.constexpr,
                     BLOCK_SIZE_ED: tl.constexpr, GROUP_SIZE_BD: tl.constexpr,
                     ORDERING: tl.constexpr = 'COLUMN_MAJOR'
                     ):
    if ORDERING == 'GROUPED':
        pid_bd, pid_ed = grouped(tl.program_id(axis=0), sample_dim, expert_dim,
                                 BLOCK_SIZE_BD, BLOCK_SIZE_ED, GROUP_SIZE_BD)
    elif ORDERING == 'COLUMN_MAJOR':
        pid_bd, pid_ed = column_major(tl.program_id(axis=0), sample_dim, BLOCK_SIZE_BD)
    elif ORDERING == 'ROW_MAJOR':
        pid_bd, pid_ed = row_major(tl.program_id(axis=0), expert_dim, BLOCK_SIZE_ED)
    expert_index = tl.program_id(axis=1)
    x_dtype = x_ptr.dtype.element_ty
    # w_dtype = weight_ptr.dtype.element_ty
    #
    expert_samples_count = tl.load(expert_bincounts_ptr + expert_index)
    bd_pids_for_expert = tl.cdiv(expert_samples_count, BLOCK_SIZE_BD)
    if pid_bd < bd_pids_for_expert:
        offs_bd = (pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)) % expert_samples_count
        offs_ed = (pid_ed * BLOCK_SIZE_ED + tl.arange(0, BLOCK_SIZE_ED)) % expert_dim
        offs_hd = tl.arange(0, BLOCK_SIZE_HD)
        # pick the data to load based on the sort indices
        in_data_indices = tl.load(sort_indices_ptr
                                  + expert_index * stride_sort_indices_ned
                                  + offs_bd * stride_sort_indices_bd).to(tl.int64)
        x_ptrs = x_ptr + \
                 in_data_indices[:, None] * stride_x_bd + \
                 offs_hd[None, :] * stride_x_hd
        w_ptrs = weight_ptr + \
                 expert_index * stride_weight_ned + \
                 offs_hd[:, None] * stride_weight_hd + \
                 offs_ed[None, :] * stride_weight_ed
        #
        # if x_dtype == tl.int8:
        #     accumulator = tl.zeros((BLOCK_SIZE_BD, BLOCK_SIZE_ED), dtype=tl.int32)
        # else:
        accumulator = tl.zeros((BLOCK_SIZE_BD, BLOCK_SIZE_ED), dtype=tl.float32)
        # potentially unoptimized
        for k in range(0, tl.cdiv(hidden_dim, BLOCK_SIZE_HD)):
            x = tl.load(x_ptrs, mask=offs_hd[None, :] < hidden_dim - k * BLOCK_SIZE_HD, other=0.0)
            w = tl.load(w_ptrs, mask=offs_hd[:, None] < hidden_dim - k * BLOCK_SIZE_HD, other=0.0)
            # if x_dtype == tl.int8:
            #     accumulator += tl.dot(x, w, out_dtype=tl.int32)
            # else:
            accumulator += tl.dot(x, w, allow_tf32=False)
            x_ptrs += BLOCK_SIZE_HD * stride_x_hd
            w_ptrs += BLOCK_SIZE_HD * stride_weight_hd
        #
        offs_b_ed = pid_ed * BLOCK_SIZE_ED + tl.arange(0, BLOCK_SIZE_ED)
        b_ptrs = bias_ptr + \
                 expert_index * stride_bias_ned + \
                 offs_b_ed[None, :] * stride_bias_ed
        # accumulator += tl.load(b_ptrs, mask=offs_b_ed[None, :] < expert_dim, other=0.0).to(tl.float32)
        accumulator += tl.load(b_ptrs, mask=offs_b_ed[None, :] < expert_dim, other=0.0)
        #
        if ACTIVATION == 'leaky_relu':
            accumulator = leaky_relu(accumulator)
        elif ACTIVATION == 'relu':
            accumulator = relu(accumulator)
        elif ACTIVATION == 'tanh':
            accumulator = tanh(accumulator)
        elif ACTIVATION == 'gelu':
            accumulator = gelu(accumulator)
        #
        out = accumulator.to(x_dtype)
        offs_out_bd = pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)
        out_ptrs = output_ptr + \
                   expert_index * stride_output_ned + \
                   offs_out_bd[:, None] * stride_output_bd + \
                   offs_b_ed[None, :] * stride_output_ed
        out_mask = (offs_out_bd[:, None] < expert_samples_count) & (offs_b_ed[None, :] < expert_dim)
        tl.store(out_ptrs, out, mask=out_mask)


class MoeSecondLayerAtomicImplementation(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, sort_indices, expert_bincounts):
        num_experts = weight.size(0)
        expert_dim = weight.size(1)
        hidden_dim = weight.size(2)
        sample_dim = input.size(1)  # 0 is axis for experts, 1 is sample, 2 is expert dimensionality
        out = torch.zeros((sample_dim, hidden_dim), device=input.device, dtype=input.dtype)
        grid = lambda META: (triton.cdiv(sample_dim, META['BLOCK_SIZE_BD']) *
                             triton.cdiv(hidden_dim, META['BLOCK_SIZE_HD']),
                             num_experts)
        moe_second_kernel_atomic[grid](input, input.stride(0), input.stride(1), input.stride(2),
                                       weight, weight.stride(0), weight.stride(1), weight.stride(2),
                                       out, out.stride(0), out.stride(1),
                                       sort_indices, sort_indices.stride(0), sort_indices.stride(1),
                                       expert_bincounts,
                                       sample_dim,
                                       expert_dim,
                                       hidden_dim,
                                       NUM_EXPERTS=num_experts,
                                       )
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        (input, weight, sort_indices, expert_bincounts) = inputs
        ctx.save_for_backward(input, weight, sort_indices, expert_bincounts)

    @staticmethod
    def backward(ctx, output_grad):
        (input, weight, sort_indices, expert_bincounts) = ctx.saved_tensors
        raise NotImplementedError('TODO')


@triton.autotune(
    configs=config_grid({
        'BLOCK_SIZE_BD': [64, 128],
        'BLOCK_SIZE_ED': [32, 64],
        'BLOCK_SIZE_HD': [64, 128, 256],
        'GROUP_SIZE_BD': [2, 4, 8, 16],
        'ORDERING': ['GROUPED']},
        num_stages=3, num_warps=8) +
            config_grid({
                'BLOCK_SIZE_BD': [32, 64, 128],
                'BLOCK_SIZE_ED': [32, 64],
                'BLOCK_SIZE_HD': [32, 64, 128, 256],
                'GROUP_SIZE_BD': [1],
                'ORDERING': ['ROW_MAJOR']},
                num_stages=4, num_warps=4) +
            config_grid({
                'BLOCK_SIZE_BD': [32, 64, 128],
                'BLOCK_SIZE_ED': [32, 64],
                'BLOCK_SIZE_HD': [32, 64, 128, 256],
                'GROUP_SIZE_BD': [1],
                'ORDERING': ['COLUMN_MAJOR']},
                num_stages=4, num_warps=4),
    # configs=[triton.Config({'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 32, 'GROUP_SIZE_BD': 1,
    #                         'ORDERING': 'COLUMN_MAJOR'}, num_stages=4, num_warps=4)],
    key=['sample_dim', 'expert_dim', 'hidden_dim', 'NUM_EXPERTS'],
)
@triton.jit
def moe_second_kernel_atomic(x_ptr, stride_x_ned, stride_x_bd, stride_x_ed,
                             weight_ptr, stride_weight_ned, stride_weight_ed, stride_weight_hd,
                             output_ptr, stride_output_bd, stride_output_hd,
                             # metadata
                             sort_indices_ptr, stride_sort_indices_bd, stride_sort_indices_ned,
                             expert_bincounts_ptr,
                             sample_dim,
                             expert_dim,
                             hidden_dim,
                             NUM_EXPERTS: tl.constexpr,
                             BLOCK_SIZE_BD: tl.constexpr, BLOCK_SIZE_HD: tl.constexpr,
                             BLOCK_SIZE_ED: tl.constexpr, GROUP_SIZE_BD: tl.constexpr,
                             ORDERING: tl.constexpr = 'GROUPED'
                             ):
    if ORDERING == 'GROUPED':
        pid_bd, pid_hd = grouped(tl.program_id(axis=0), sample_dim, hidden_dim,
                                 BLOCK_SIZE_BD, BLOCK_SIZE_HD, GROUP_SIZE_BD)
    elif ORDERING == 'COLUMN_MAJOR':
        pid_bd, pid_hd = column_major(tl.program_id(axis=0), sample_dim, BLOCK_SIZE_BD)
    elif ORDERING == 'ROW_MAJOR':
        pid_bd, pid_hd = row_major(tl.program_id(axis=0), hidden_dim, BLOCK_SIZE_HD)
    expert_index = tl.program_id(axis=1)
    x_dtype = x_ptr.dtype.element_ty
    # w_dtype = weight_ptr.dtype.element_ty
    #
    expert_samples_count = tl.load(expert_bincounts_ptr + expert_index)
    bd_pids_for_expert = tl.cdiv(expert_samples_count, BLOCK_SIZE_BD)
    if pid_bd < bd_pids_for_expert:
        offs_bd = (pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)) % expert_samples_count
        offs_hd = (pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)) % hidden_dim
        offs_ed = tl.arange(0, BLOCK_SIZE_ED)
        x_ptrs = x_ptr + \
                 expert_index * stride_x_ned + \
                 offs_bd[:, None] * stride_x_bd + \
                 offs_ed[None, :] * stride_x_ed
        w_ptrs = weight_ptr + \
                 expert_index * stride_weight_ned + \
                 offs_ed[:, None] * stride_weight_ed + \
                 offs_hd[None, :] * stride_weight_hd
        accumulator = tl.zeros((BLOCK_SIZE_BD, BLOCK_SIZE_HD), dtype=tl.float32)
        for k in range(0, tl.cdiv(expert_dim, BLOCK_SIZE_ED)):
            x = tl.load(x_ptrs, mask=offs_ed[None, :] < expert_dim - k * BLOCK_SIZE_ED, other=0.0)
            w = tl.load(w_ptrs, mask=offs_ed[:, None] < expert_dim - k * BLOCK_SIZE_ED, other=0.0).to(x_dtype)
            accumulator += tl.dot(x, w, allow_tf32=False)
            x_ptrs += BLOCK_SIZE_ED * stride_x_ed
            w_ptrs += BLOCK_SIZE_ED * stride_weight_ed
        out_data_indices = tl.load(sort_indices_ptr
                                   + expert_index * stride_sort_indices_ned
                                   + offs_bd * stride_sort_indices_bd).to(tl.int64)
        offs_out_bd = pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)
        offs_out_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)
        out_ptrs = output_ptr + \
                   out_data_indices[:, None] * stride_output_bd + \
                   offs_out_hd[None, :] * stride_output_hd
        out_mask = (offs_out_bd[:, None] < expert_samples_count) & (offs_out_hd[None, :] < hidden_dim)
        out = accumulator.to(x_dtype)
        tl.atomic_add(out_ptrs, out, mask=out_mask, sem='relaxed', scope='gpu')


class MoeSecondLayerMergingImplementation(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, unsort_indices, expert_bincounts, routing_tensor):
        num_experts = weight.size(0)
        expert_dim = weight.size(1)
        hidden_dim = weight.size(2)
        sample_dim = input.size(1)  # 0 is axis for experts, 1 is sample, 2 is expert dimensionality
        intermediate_out = torch.empty((num_experts, sample_dim, hidden_dim), device=input.device, dtype=input.dtype)
        grid = lambda META: (triton.cdiv(sample_dim, META['BLOCK_SIZE_BD']) *
                             triton.cdiv(hidden_dim, META['BLOCK_SIZE_HD']),
                             num_experts)
        # assert weight.stride(2) == 1, f'{weight.stride(2)=}'
        # assert weight.stride(1) > 1, f'{weight.stride(1)=}'
        # if input.dtype == torch.int8:
        #     # tensor cores require column-major ordering for optimal performance
        #     # for B/weight matrix when using INT8 etc.
        #     # so we temporarily switch the strides for benchmarking purposes
        #     moe_second_kernel[grid](input, input.stride(0), input.stride(1), input.stride(2),
        #                             weight, weight.stride(0), weight.stride(2), weight.stride(1),
        #                             intermediate_out, intermediate_out.stride(0),
        #                             intermediate_out.stride(1), intermediate_out.stride(2),
        #                             expert_bincounts,
        #                             sample_dim,
        #                             expert_dim,
        #                             hidden_dim,
        #                             NUM_EXPERTS=num_experts,
        #                             )
        # else:
        moe_second_kernel[grid](input, input.stride(0), input.stride(1), input.stride(2),
                                weight, weight.stride(0), weight.stride(1), weight.stride(2),
                                intermediate_out, intermediate_out.stride(0),
                                intermediate_out.stride(1), intermediate_out.stride(2),
                                expert_bincounts,
                                sample_dim,
                                expert_dim,
                                hidden_dim,
                                NUM_EXPERTS=num_experts,
                                )
        out = torch.empty((sample_dim, hidden_dim), device=input.device, dtype=input.dtype)
        merge_grid = lambda META: (sample_dim, triton.cdiv(hidden_dim, META['BLOCK_SIZE_HD']))
        moe_merge_results_kernel[merge_grid](intermediate_out, intermediate_out.stride(0),
                                             intermediate_out.stride(1), intermediate_out.stride(2),
                                             out, out.stride(0), out.stride(1),
                                             unsort_indices, unsort_indices.stride(0), unsort_indices.stride(1),
                                             routing_tensor, routing_tensor.stride(0), routing_tensor.stride(1),
                                             hidden_dim,
                                             NUM_EXPERTS=num_experts)
        # merge_grid = lambda META: (triton.cdiv(sample_dim, META['BLOCK_SIZE_BD']),
        #                            triton.cdiv(hidden_dim, META['BLOCK_SIZE_HD']))
        # moe_merge_results_batched_kernel[merge_grid](intermediate_out, intermediate_out.stride(0),
        #                                              intermediate_out.stride(1), intermediate_out.stride(2),
        #                                              out, out.stride(0), out.stride(1),
        #                                              unsort_indices, unsort_indices.stride(0),
        #                                              unsort_indices.stride(1),
        #                                              routing_tensor, routing_tensor.stride(0),
        #                                              routing_tensor.stride(1),
        #                                              sample_dim,
        #                                              hidden_dim,
        #                                              NUM_EXPERTS=num_experts)
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        (input, weight, sort_indices, expert_bincounts, routing_tensor) = inputs
        ctx.save_for_backward(input, weight, sort_indices, expert_bincounts, routing_tensor)

    @staticmethod
    def backward(ctx, output_grad):
        (input, weight, sort_indices, expert_bincounts, routing_tensor) = ctx.saved_tensors
        raise NotImplementedError('TODO')


@triton.autotune(
    configs=config_grid({
        'BLOCK_SIZE_BD': [64, 128],
        'BLOCK_SIZE_ED': [32, 64],
        'BLOCK_SIZE_HD': [64, 128, 256],
        'GROUP_SIZE_BD': [4, 8, 16],
        'ORDERING': ['GROUPED']},
        num_stages=3, num_warps=8) +
            config_grid({
                'BLOCK_SIZE_BD': [32, 64, 128],
                'BLOCK_SIZE_ED': [32, 64, 128],
                'BLOCK_SIZE_HD': [32, 64, 128, 256],
                'GROUP_SIZE_BD': [1],
                'ORDERING': ['ROW_MAJOR']},
                num_stages=4, num_warps=4) +
            config_grid({
                'BLOCK_SIZE_BD': [32, 64, 128],
                'BLOCK_SIZE_ED': [32, 64, 128],
                'BLOCK_SIZE_HD': [32, 64, 128, 256],
                'GROUP_SIZE_BD': [1],
                'ORDERING': ['COLUMN_MAJOR']},
                num_stages=4, num_warps=4),
    # configs=[triton.Config({'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_ED': 64, 'BLOCK_SIZE_HD': 128, 'GROUP_SIZE_BD': 1,
    #                         'ORDERING': 'COLUMN_MAJOR'}, num_stages=4, num_warps=4)],
    key=['sample_dim', 'expert_dim', 'hidden_dim', 'NUM_EXPERTS'],
)
@triton.jit
def moe_second_kernel(x_ptr, stride_x_ned, stride_x_bd, stride_x_ed,
                      weight_ptr, stride_weight_ned, stride_weight_ed, stride_weight_hd,
                      output_ptr, stride_output_ned, stride_output_bd, stride_output_hd,
                      # metadata
                      expert_bincounts_ptr,
                      sample_dim,
                      expert_dim,
                      hidden_dim,
                      NUM_EXPERTS: tl.constexpr,
                      BLOCK_SIZE_BD: tl.constexpr, BLOCK_SIZE_HD: tl.constexpr,
                      BLOCK_SIZE_ED: tl.constexpr, GROUP_SIZE_BD: tl.constexpr,
                      ORDERING: tl.constexpr = 'GROUPED'
                      ):
    if ORDERING == 'GROUPED':
        pid_bd, pid_hd = grouped(tl.program_id(axis=0), sample_dim, hidden_dim,
                                 BLOCK_SIZE_BD, BLOCK_SIZE_HD, GROUP_SIZE_BD)
    elif ORDERING == 'COLUMN_MAJOR':
        pid_bd, pid_hd = column_major(tl.program_id(axis=0), sample_dim, BLOCK_SIZE_BD)
    elif ORDERING == 'ROW_MAJOR':
        pid_bd, pid_hd = row_major(tl.program_id(axis=0), hidden_dim, BLOCK_SIZE_HD)
    expert_index = tl.program_id(axis=1)
    x_dtype = x_ptr.dtype.element_ty
    # w_dtype = weight_ptr.dtype.element_ty
    #
    expert_samples_count = tl.load(expert_bincounts_ptr + expert_index)
    bd_pids_for_expert = tl.cdiv(expert_samples_count, BLOCK_SIZE_BD)
    if pid_bd < bd_pids_for_expert:
        offs_bd = (pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)) % expert_samples_count
        offs_hd = (pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)) % hidden_dim
        offs_ed = tl.arange(0, BLOCK_SIZE_ED)
        x_ptrs = x_ptr + \
                 expert_index * stride_x_ned + \
                 offs_bd[:, None] * stride_x_bd + \
                 offs_ed[None, :] * stride_x_ed
        w_ptrs = weight_ptr + \
                 expert_index * stride_weight_ned + \
                 offs_ed[:, None] * stride_weight_ed + \
                 offs_hd[None, :] * stride_weight_hd
        # if x_dtype == tl.int8:
        #     accumulator = tl.zeros((BLOCK_SIZE_BD, BLOCK_SIZE_HD), dtype=tl.int32)
        # else:
        accumulator = tl.zeros((BLOCK_SIZE_BD, BLOCK_SIZE_HD), dtype=tl.float32)
        for k in range(0, tl.cdiv(expert_dim, BLOCK_SIZE_ED)):
            x = tl.load(x_ptrs, mask=offs_ed[None, :] < expert_dim - k * BLOCK_SIZE_ED, other=0.0)
            w = tl.load(w_ptrs, mask=offs_ed[:, None] < expert_dim - k * BLOCK_SIZE_ED, other=0.0)
            # if x_dtype == tl.int8:
            #     accumulator += tl.dot(x, w, out_dtype=tl.int32)
            # else:
            #     accumulator = tl.zeros((BLOCK_SIZE_BD, BLOCK_SIZE_HD), dtype=tl.float32)
            accumulator += tl.dot(x, w, allow_tf32=False)
            x_ptrs += BLOCK_SIZE_ED * stride_x_ed
            w_ptrs += BLOCK_SIZE_ED * stride_weight_ed
        offs_out_bd = pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)
        offs_out_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)
        out_ptrs = output_ptr + \
                   expert_index * stride_output_ned + \
                   offs_out_bd[:, None] * stride_output_bd + \
                   offs_out_hd[None, :] * stride_output_hd
        out_mask = (offs_out_bd[:, None] < expert_samples_count) & (offs_out_hd[None, :] < hidden_dim)
        out = accumulator.to(x_dtype)
        tl.store(out_ptrs, out, mask=out_mask)


@triton.autotune(
    configs=config_grid({
        'BLOCK_SIZE_HD': [64, 128, 256], }, num_stages=4, num_warps=4),
    # configs=[triton.Config({
    #     'BLOCK_SIZE_HD': 256}, num_stages=4, num_warps=4)],
    key=['hidden_dim', 'NUM_EXPERTS'],
)
@triton.jit
def moe_merge_results_kernel(x_ptr, stride_x_ned, stride_x_bd, stride_x_hd,
                             output_ptr, stride_output_bd, stride_output_hd,
                             # metadata
                             unsort_indices_ptr, stride_unsort_indices_bd, stride_unsort_indices_ned,
                             routing_tensor_ptr, stride_routing_tensor_bd, stride_routing_tensor_ned,
                             hidden_dim,
                             NUM_EXPERTS: tl.constexpr,
                             BLOCK_SIZE_HD: tl.constexpr
                             ):
    # TODO optimize by vectorizing
    pid_bd = tl.program_id(axis=0)
    pid_hd = tl.program_id(axis=1)
    offs_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)
    hd_mask = offs_hd < hidden_dim
    x_dtype = x_ptr.dtype.element_ty
    # TODO optimize after indexing becomes available in Triton
    # https://github.com/triton-lang/triton/issues/974
    # offs_ned = tl.arange(0, NUM_EXPERTS)
    # routing_info = tl.load(routing_tensor_ptr +
    #                        pid_bd * stride_routing_tensor_bd +
    #                        offs_ned * stride_routing_tensor_ned)
    # if x_dtype == tl.int8:
    #     accumulator = tl.zeros((BLOCK_SIZE_HD,), dtype=tl.int32)
    # else:
    accumulator = tl.zeros((BLOCK_SIZE_HD,), dtype=tl.float32)
    for i in range(0, NUM_EXPERTS):
        # inefficient uncoalesced load - TODO optimize?
        expert_executed = tl.load(routing_tensor_ptr +
                                  pid_bd * stride_routing_tensor_bd +
                                  i * stride_routing_tensor_ned)
        if expert_executed > 0:
            sample_index = tl.load(unsort_indices_ptr +
                                   pid_bd * stride_unsort_indices_bd +
                                   i * stride_unsort_indices_ned)
            accumulator += tl.load(x_ptr +
                                   i * stride_x_ned +
                                   sample_index * stride_x_bd +
                                   offs_hd * stride_x_hd)
    out_ptrs = output_ptr + pid_bd * stride_output_bd + offs_hd * stride_output_hd
    tl.store(out_ptrs, accumulator.to(x_dtype), mask=hd_mask)


@triton.autotune(
    configs=config_grid({
        'BLOCK_SIZE_BD': [32, 64, 128],
        'BLOCK_SIZE_HD': [32, 64, 128], }, num_stages=4, num_warps=4),
    # configs=[triton.Config({
    #     'BLOCK_SIZE_BD': 64,
    #     'BLOCK_SIZE_HD': 64}, num_stages=4, num_warps=4)],
    key=['hidden_dim', 'NUM_EXPERTS'],
)
@triton.jit
def moe_merge_results_batched_kernel(x_ptr, stride_x_ned, stride_x_bd, stride_x_hd,
                                     output_ptr, stride_output_bd, stride_output_hd,
                                     # metadata
                                     unsort_indices_ptr, stride_unsort_indices_bd, stride_unsort_indices_ned,
                                     routing_tensor_ptr, stride_routing_tensor_bd, stride_routing_tensor_ned,
                                     sample_dim,
                                     hidden_dim,
                                     NUM_EXPERTS: tl.constexpr,
                                     BLOCK_SIZE_BD: tl.constexpr,
                                     BLOCK_SIZE_HD: tl.constexpr
                                     ):
    pid_bd = tl.program_id(axis=0)
    pid_hd = tl.program_id(axis=1)
    offs_bd = pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)
    offs_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)
    # offs_ned = tl.arange(0, NUM_EXPERTS)
    bd_mask = offs_bd < sample_dim
    hd_mask = offs_hd < hidden_dim
    x_dtype = x_ptr.dtype.element_ty
    # TODO vectorize/optimize
    # experts_executed = tl.load(routing_tensor_ptr +
    #                            offs_bd[:, None] * stride_routing_tensor_bd +
    #                            offs_ned[None, :] * stride_routing_tensor_ned)
    accumulator = tl.zeros((BLOCK_SIZE_BD, BLOCK_SIZE_HD), dtype=tl.float32)
    for i in range(0, NUM_EXPERTS):
        # inefficient uncoalesced load - TODO optimize?
        samples_executed = tl.load(routing_tensor_ptr +
                                   offs_bd * stride_routing_tensor_bd +
                                   i * stride_routing_tensor_ned,
                                   mask=bd_mask)
        samples_executed_mask = (samples_executed > 0) & (bd_mask)
        sample_indices = tl.load(unsort_indices_ptr +
                                 offs_bd * stride_unsort_indices_bd +
                                 i * stride_unsort_indices_ned,
                                 mask=samples_executed_mask)
        load_mask = samples_executed_mask[:, None] & hd_mask[None, :]
        accumulator += tl.load(x_ptr +
                               i * stride_x_ned +
                               sample_indices[:, None] * stride_x_bd +
                               offs_hd[None, :] * stride_x_hd,
                               mask=load_mask)
    out_ptrs = output_ptr + offs_bd[:, None] * stride_output_bd + offs_hd[None, :] * stride_output_hd
    out_mask = bd_mask[:, None] & hd_mask[None, :]
    tl.store(out_ptrs, accumulator.to(x_dtype), mask=out_mask)
