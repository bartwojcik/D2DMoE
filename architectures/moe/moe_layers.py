import math

import torch
import torch.nn as nn

from architectures.moe.moe_impls import MoeFirstLayerImplementation, MoeSecondLayerAtomicImplementation, \
    MoeSecondLayerMergingImplementation

# TODO change "activation" arguments to use ACTIVATION_NAME_MAP?
# TODO refactor
ACTIVATION_TO_NAME = {
    nn.GELU: 'gelu',
    nn.Tanh: 'tanh',
    nn.ReLU: 'relu',
    nn.LeakyReLU: 'leaky_relu',
}


def init_(p):
    with torch.no_grad():
        dim = p.size(-1)
        std = 1 / math.sqrt(dim)
        p.uniform_(-std, std)
    return p


class ExpertsLayer(nn.Module):
    def __init__(self):
        super().__init__()


class ExecuteAllExpertsLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, bias, activation):
        super().__init__()
        self.bias = bias
        self.w = nn.Parameter(torch.empty(num_experts, in_dim, out_dim))
        init_(self.w)
        if self.bias:
            self.b = nn.Parameter(torch.empty(num_experts, 1, out_dim))
            init_(self.b)
        self.act = activation()

    def forward(self, x):
        x = torch.einsum('eni,eio->eno', x, self.w)
        if self.bias:
            x = x + self.b
        return self.act(x)


class ExecuteAllExperts(ExpertsLayer):
    def __init__(self,
                 dim,
                 num_experts,
                 depth=2,
                 expert_dim=None,
                 bias=True,
                 activation=nn.GELU,
                 intermediate_gating=False):
        super().__init__()
        assert depth >= 2
        self.num_experts = num_experts
        self.depth = depth
        self.dim = dim
        self.expert_dim = dim * 2 if expert_dim is None else expert_dim
        self.bias_mode = bias
        self.activation = activation
        # assumes homogeneous experts
        self.layers = nn.ModuleList()
        bias = True if self.bias_mode == 'without_last' else self.bias_mode
        self.intermediate_gating = intermediate_gating
        if intermediate_gating:
            if depth > 2:
                raise NotImplementedError('Intermediate gating not supported for depth > 2')
            # LLaMa - like intermediate gating
            # We assume layer 0 is the gate, 1 is the output and 2 is the up-projection
            self.layers.append(ExecuteAllExpertsLayer(dim, expert_dim, num_experts, bias, activation))
            self.layers.append(ExecuteAllExpertsLayer(expert_dim, dim, num_experts, bias, nn.Identity))
            self.layers.append(ExecuteAllExpertsLayer(dim, expert_dim, num_experts, bias, nn.Identity))
        else:
            self.layers.append(ExecuteAllExpertsLayer(dim, expert_dim, num_experts, bias, activation))
            for _ in range(depth - 2):
                self.layers.append(ExecuteAllExpertsLayer(expert_dim, expert_dim, num_experts, bias, activation))
            bias = False if self.bias_mode == 'without_last' else self.bias_mode
            self.layers.append(ExecuteAllExpertsLayer(expert_dim, dim, num_experts, bias, nn.Identity))

    def forward(self, x, routing_tensor):
        assert x.dim() == 2, f'{x.size()=}'
        # x is of size (batch_size * sequence_length, dim)
        # routing_tensor is of size (batch_size * sequence_length, num_experts)
        x = x.unsqueeze(0)
        if self.intermediate_gating:
            # Apply gated activation in the middle
            x = self.layers[0](x) * self.layers[2](x)
            x = self.layers[1](x)
            x = torch.einsum('end,ne->nd', x, routing_tensor)
            return x
        else:
            for layer in self.layers:
                x = layer(x)
            x = torch.einsum('end,ne->nd', x, routing_tensor)
            return x


class GLUExpert(nn.Module):
    def __init__(self, d_model, d_intermediate, activation, bias, last_bias):
        super().__init__()
        self.gate = nn.Linear(d_model, d_intermediate, bias=bias)
        self.up_scale = nn.Linear(d_model, d_intermediate, bias=bias)
        self.activation = activation()
        self.down_scale = nn.Linear(d_intermediate, d_model, bias=last_bias)

    def forward(self, x):
        return self.down_scale(self.activation(self.gate(x)) * self.up_scale(x))


class ExpertIntermediateActivationExtractionStub(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w1, b1, act):
        x = torch.einsum('eni,eio->eno', x, w1)
        x = x + b1.unsqueeze(-2)
        x = act(x)
        return x


class ExpertOutputActivationExtractionStub(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w2):
        x = torch.einsum('eni,eio->eno', x, w2)
        return x


class CustomKernelExperts(ExpertsLayer):
    def __init__(self,
                 dim,
                 num_experts,
                 depth=2,
                 expert_dim=None,
                 bias=True,
                 activation=nn.GELU,
                 intermediate_gating=False):
        if intermediate_gating:
            raise NotImplementedError('intermediate_gating not supported yet')
        super().__init__()
        assert depth == 2, f'No support for depth != 2 for now'
        assert bias == 'without_last', f'No support for bias != "without_last" for now'
        assert activation in ACTIVATION_TO_NAME, f'{activation} not supported by the kernel'
        self.num_experts = num_experts
        self.depth = depth
        self.expert_dim = dim * 2 if expert_dim is None else expert_dim
        self.hidden_dim = dim
        self.bias_mode = bias
        # assumes homogeneous experts
        self.w1 = init_(nn.Parameter(torch.empty(num_experts, dim, expert_dim)))
        self.b1 = init_(nn.Parameter(torch.empty(num_experts, expert_dim)))
        self.w2 = init_(nn.Parameter(torch.empty(num_experts, expert_dim, dim)))
        self.act = activation()
        self.activation_str = ACTIVATION_TO_NAME[activation]
        self._forward_mode = 'masked'
        self.intermediate_activation_extraction_stub = ExpertIntermediateActivationExtractionStub()
        self.output_activation_extraction_stub = ExpertOutputActivationExtractionStub()

    @property
    def forward_mode(self):
        return self._forward_mode

    @forward_mode.setter
    def forward_mode(self, value):
        self._forward_mode = value

    def forward(self, x, routing_tensor):
        assert x.dim() == 2, f'{x.size()=}'
        # x is of size (batch_size * sequence_length, dim)
        # routing_tensor is of size (batch_size * sequence_length, num_experts)
        if self._forward_mode == 'masked':
            res = self.forward_all(x, routing_tensor)
        elif self._forward_mode == 'naive':
            res = self.forward_naive(x, routing_tensor)
        elif self._forward_mode == 'triton':
            res = self.forward_triton(x, routing_tensor)
        elif self._forward_mode == 'triton_atomic':
            res = self.forward_triton_atomic(x, routing_tensor)
        elif self._forward_mode == 'triton_optimized':
            res = self.forward_triton_optimized(x, routing_tensor)
        else:
            raise NotImplementedError(f'Unsupported forward_mode: {self._forward_mode}')
        # print(f'{self._forward_mode=} {res=}')
        return res

    def forward_all(self, x, routing_tensor):
        x = x.unsqueeze(0)
        x = self.intermediate_activation_extraction_stub(x, self.w1, self.b1, self.act)
        x = self.output_activation_extraction_stub(x, self.w2)
        x = torch.einsum('end,ne->nd', x, routing_tensor)
        return x

    def forward_naive(self, x, routing_tensor):
        outputs = torch.zeros_like(x)
        for i in range(self.num_experts):
            current_expert_routing_tensor = routing_tensor[:, i]
            current_expert_samples_mask = current_expert_routing_tensor > 0.0
            # current_expert_samples_scores = current_expert_routing_tensor[current_expert_samples_mask].unsqueeze(1)
            # assert current_expert_samples_scores.dim() == 2, f'{current_expert_samples_scores.size()=}'
            current_expert_x = x[current_expert_samples_mask]
            current_expert_x = self.act(current_expert_x @ self.w1[i] + self.b1[i])
            current_expert_x = current_expert_x @ self.w2[i]
            outputs[current_expert_samples_mask] += current_expert_x
        return outputs

    @staticmethod
    # @torch.compile(fullgraph=True, dynamic=True, mode='max-autotune')
    @torch.jit.script
    def extract_indices(routing_tensor):
        # TODO is casting necessary?
        routing_tensor = routing_tensor.to(torch.int32)
        sort_indices = routing_tensor.argsort(dim=0, descending=True)
        expert_bincounts = routing_tensor.sum(dim=0)
        # samples_cumsum_by_expert = routing_tensor.cumsum(dim=0)
        # expert_bincounts, _ = samples_cumsum_by_expert.max(dim=0)
        unsort_indices = sort_indices.argsort(dim=0)
        return sort_indices, unsort_indices, expert_bincounts

    def forward_triton(self, x, routing_tensor):
        with torch.no_grad():
            sort_indices, unsort_indices, expert_bincounts = self.extract_indices(routing_tensor)
        intermediate_acts = MoeFirstLayerImplementation.apply(x, self.w1, self.b1,
                                                              sort_indices,
                                                              expert_bincounts,
                                                              self.activation_str)
        final_out = MoeSecondLayerMergingImplementation.apply(intermediate_acts, self.w2,
                                                              unsort_indices,
                                                              expert_bincounts,
                                                              routing_tensor)
        return final_out

    # torch.compile causes synchronizations
    # TODO check if this is valid for newer pytorch versions
    @staticmethod
    # @torch.compile(fullgraph=True, dynamic=True, mode='max-autotune')
    @torch.jit.script
    def extract_indices_atomic(routing_tensor):
        # TODO is casting necessary?
        routing_tensor = routing_tensor.to(torch.int32)
        sort_indices = routing_tensor.argsort(dim=0, descending=True)
        expert_bincounts = routing_tensor.sum(dim=0)
        return sort_indices, expert_bincounts

    def forward_triton_atomic(self, x, routing_tensor):
        with torch.no_grad():
            sort_indices, expert_bincounts = \
                self.extract_indices_atomic(routing_tensor)
        intermediate_acts = MoeFirstLayerImplementation.apply(
            x, self.w1, self.b1,
            sort_indices,
            expert_bincounts,
            self.activation_str)
        final_out = MoeSecondLayerAtomicImplementation.apply(
            intermediate_acts, self.w2,
            sort_indices,
            expert_bincounts)
        return final_out


class ModuleBatchedExperts(ExpertsLayer):
    def __init__(self,
                 dim,
                 num_experts,
                 depth=2,
                 expert_dim=None,
                 bias=True,
                 activation=nn.GELU,
                 intermediate_gating=False):
        super().__init__()
        assert depth >= 2
        self.num_experts = num_experts
        self.depth = depth
        self.expert_dim = dim * 2 if expert_dim is None else expert_dim
        self.bias_mode = bias
        self.intermediate_gating = intermediate_gating
        if intermediate_gating and depth > 2:
            raise NotImplementedError('Intermediate gating not supported for depth > 2')

        self.e = nn.ModuleList()
        for _ in range(num_experts):
            if self.intermediate_gating:
                bias = True if self.bias_mode != False else False
                last_bias = False if (self.bias_mode == False or self.bias_mode == 'without_last') else True
                current_expert = GLUExpert(
                    dim,
                    self.expert_dim,
                    activation,
                    bias,
                    last_bias
                )
            else:
                current_expert = torch.nn.Sequential()
                bias = True if self.bias_mode == 'without_last' else self.bias_mode
                current_expert.append(nn.Linear(dim, expert_dim, bias=bias))
                current_expert.append(activation())
                for _ in range(depth - 2):
                    current_expert.append(nn.Linear(expert_dim, expert_dim, bias=bias))
                    current_expert.append(activation())
                bias = False if self.bias_mode == 'without_last' else self.bias_mode
                current_expert.append(nn.Linear(expert_dim, dim, bias=bias))
            self.e.append(current_expert)

    def forward(self, x, routing_tensor):
        assert not torch.any(torch.isnan(x)), f'{x=}'
        assert x.dim() == 2, f'{x.size()=}'
        # x is of size (batch_size * sequence_length, dim)
        # routing_tensor is of size (batch_size * sequence_length, num_experts)
        outputs = torch.zeros_like(x)
        for i in range(self.num_experts):
            current_expert_routing_tensor = routing_tensor[:, i]
            current_expert_samples_mask = current_expert_routing_tensor != 0.0
            current_expert_samples_scores = current_expert_routing_tensor[current_expert_samples_mask].unsqueeze(1)
            current_expert_x = x[current_expert_samples_mask]
            assert current_expert_samples_scores.dim() == 2, f'{current_expert_samples_scores.size()=}'
            current_expert_x = self.e[i](current_expert_x) * current_expert_samples_scores
            outputs[current_expert_samples_mask] += current_expert_x
        return outputs


class BatchedExperts(ExpertsLayer):
    def __init__(self,
                 dim,
                 num_experts,
                 depth=2,
                 expert_dim=None,
                 bias=True,
                 activation=nn.GELU,
                 intermediate_gating=False):
        super().__init__()
        assert depth >= 2
        self.num_experts = num_experts
        self.depth = depth
        self.expert_dim = dim * 2 if expert_dim is None else expert_dim
        self.bias_mode = bias
        bias = True if self.bias_mode == 'without_last' else self.bias_mode
        if intermediate_gating:
            raise NotImplementedError()
        # assumes homogeneous experts
        self.weights = nn.ParameterList()
        if bias:
            self.biases = nn.ParameterList()
        # add weights for the first layer
        w = init_(torch.empty(num_experts, dim, expert_dim))
        self.weights.append(w)
        if bias:
            self.biases.append(torch.zeros(num_experts, 1, expert_dim))
        # add weights for the intermediate layers
        for _ in range(depth - 2):
            w = init_(torch.zeros(num_experts, expert_dim, expert_dim))
            self.weights.append(w)
            if bias:
                self.biases.append(torch.zeros(num_experts, 1, expert_dim))
        # add weights for the last layer
        w = init_(torch.zeros(num_experts, expert_dim, dim))
        self.weights.append(w)
        bias = True if self.bias_mode == 'without_last' else self.bias_mode
        if bias:
            self.biases.append(torch.zeros(num_experts, 1, dim))
        self.act = activation()

    def expert_forward(self, x: torch.Tensor, expert_index: int):
        # x is of size (batch_size * sequence_length, dim)
        # expert index is an int
        if self.bias_mode is True:
            for i in range(0, self.depth - 1):
                x = self.act(x @ self.weights[i][expert_index] + self.biases[i][expert_index])
            x = x @ self.weights[self.depth - 1][expert_index] + self.biases[self.depth - 1][expert_index]
        elif self.bias_mode == 'without_last':
            for i in range(0, self.depth - 1):
                x = self.act(x @ self.weights[i][expert_index] + self.biases[i][expert_index])
            x = x @ self.weights[self.depth - 1][expert_index]
        else:
            for i in range(0, self.depth - 1):
                x = self.act(x @ self.weights[i][expert_index])
            x = x @ self.weights[self.depth - 1][expert_index]
        return x

    def forward(self, x, routing_tensor):
        assert not torch.any(torch.isnan(x)), f'{x=}'
        assert x.dim() == 2, f'{x.size()=}'
        # x is of size (batch_size * sequence_length, dim)
        # routing_tensor is of size (batch_size * sequence_length, num_experts)
        outputs = torch.zeros_like(x)
        for i in range(self.num_experts):
            current_expert_routing_tensor = routing_tensor[:, i]
            current_expert_samples_mask = current_expert_routing_tensor != 0.0
            current_expert_samples_scores = current_expert_routing_tensor[current_expert_samples_mask].unsqueeze(1)
            current_expert_x = x[current_expert_samples_mask]
            assert current_expert_samples_scores.dim() == 2, f'{current_expert_samples_scores.size()=}'
            current_expert_x = self.expert_forward(current_expert_x, i) * current_expert_samples_scores
            outputs[current_expert_samples_mask] += current_expert_x
        return outputs


MOE_IMPL_MAP = {
    'execute_all': ExecuteAllExperts,
    'module': ModuleBatchedExperts,
    'batched': BatchedExperts,
    'custom_kernel': CustomKernelExperts,
}


class MoELayer(nn.Module):
    def __init__(self, name, num_experts):
        super().__init__()
        self.name = name
        self.num_experts = num_experts
