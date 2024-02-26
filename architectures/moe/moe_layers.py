import math

import torch
import torch.nn as nn

from utils import gumbel_sigmoid


class ExpertsLayer(nn.Module):
    def __init__(self):
        super().__init__()


class ExecuteAllExpertsLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, bias, activation):
        super().__init__()
        self.bias = bias
        self.w = nn.Parameter(torch.zeros(num_experts, in_dim, out_dim))
        self.init_(self.w)
        if self.bias:
            self.b = nn.Parameter(torch.zeros(num_experts, 1, out_dim))
            self.init_(self.b)
        self.act = activation()

    @staticmethod
    def init_(t):
        with torch.no_grad():
            dim = t.size(-1)
            std = 1 / math.sqrt(dim)
            t.uniform_(-std, std)

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
                 activation=nn.GELU):
        super().__init__()
        assert depth >= 2
        self.num_experts = num_experts
        self.depth = depth
        self.expert_dim = dim * 2 if expert_dim is None else expert_dim
        self.bias_mode = bias
        # assumes homogeneous experts
        self.layers = nn.ModuleList()
        bias = True if self.bias_mode == 'without_last' else self.bias_mode
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
        for layer in self.layers:
            x = layer(x)
        x = torch.einsum('end,ne->nd', x, routing_tensor)
        return x


class ModuleBatchedExperts(ExpertsLayer):
    def __init__(self,
                 dim,
                 num_experts,
                 depth=2,
                 expert_dim=None,
                 bias=True,
                 activation=nn.GELU):
        super().__init__()
        assert depth >= 2
        self.num_experts = num_experts
        self.depth = depth
        self.expert_dim = dim * 2 if expert_dim is None else expert_dim
        self.bias_mode = bias

        self.e = nn.ModuleList()
        for _ in range(num_experts):
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
                 activation=nn.GELU):
        super().__init__()
        assert depth >= 2
        self.num_experts = num_experts
        self.depth = depth
        self.expert_dim = dim * 2 if expert_dim is None else expert_dim
        self.bias_mode = bias
        bias = True if self.bias_mode == 'without_last' else self.bias_mode
        # assumes homogeneous experts
        self.weights = nn.ParameterList()
        if bias:
            self.biases = nn.ParameterList()
        # add weights for the first layer
        w = torch.zeros(num_experts, dim, expert_dim)
        self.init_(w)
        self.weights.append(w)
        if bias:
            self.biases.append(torch.zeros(num_experts, 1, expert_dim))
        # add weights for the intermediate layers
        for _ in range(depth - 2):
            w = torch.zeros(num_experts, expert_dim, expert_dim)
            self.init_(w)
            self.weights.append(w)
            if bias:
                self.biases.append(torch.zeros(num_experts, 1, expert_dim))
        # add weights for the last layer
        w = torch.zeros(num_experts, expert_dim, dim)
        self.init_(w)
        self.weights.append(w)
        bias = True if self.bias_mode == 'without_last' else self.bias_mode
        if bias:
            self.biases.append(torch.zeros(num_experts, 1, dim))
        self.act = activation()

    @staticmethod
    def init_(t):
        dim = t.size(-1)
        std = 1 / math.sqrt(dim)
        t.uniform_(-std, std)

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
}


class MoELayer(nn.Module):
    def __init__(self, name, num_experts):
        super().__init__()
        self.name = name
        self.num_experts = num_experts