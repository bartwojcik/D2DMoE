import math

import torch
from torch import nn

from architectures.gpt import CausalSelfAttention
from utils import find_module_names, get_module_by_name, set_module_by_name, get_module_name, get_parent_module_name


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, 'Embedding dimension must be divisible by the number of heads.'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout_p
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.register_buffer(
            "causal_mask",
            None
        )

    def scaled_dot_product(self, q, k, v, dropout_p=0.0):
        d_k = q.size(-1)
        seq_len = q.size(-2)
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if self.causal_mask is not None:
            # TODO check for sure that seq_len is correct dimension here
            attn_logits = attn_logits.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        attention = nn.functional.softmax(attn_logits, dim=-1)
        if dropout_p > 0.0:
            attention = nn.functional.dropout(attention, p=dropout_p)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, query, key, value, _key_padding_mask=None, need_weights=False):
        assert query.size() == key.size() == value.size()
        batch_size, seq_length, embed_dim = query.size()  # [B, T, E]
        query, key, value = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        # separate the head dimension, and permute dimensions into [Batch, Head, SeqLen, Dims]
        query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # Determine value outputs
        values, attention = self.scaled_dot_product(query, key, value,
                                                    dropout_p=self.dropout_p if self.training else 0.0)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        if need_weights:
            return o, attention
        else:
            return o, None


def transcribe_mha_params(org_mha, simple_mha):
    if isinstance(org_mha, nn.MultiheadAttention):
        assert org_mha._qkv_same_embed_dim is True
        assert org_mha.embed_dim == simple_mha.embed_dim
        assert org_mha.num_heads == simple_mha.num_heads
        assert org_mha.in_proj_bias is not None and org_mha.bias_v is None and org_mha.bias_k is None
        assert org_mha.out_proj.bias is not None
        assert org_mha.batch_first is True
        embed_dim = org_mha.embed_dim
        simple_mha.q_proj.weight = torch.nn.Parameter(org_mha.in_proj_weight[:embed_dim].clone().detach())
        simple_mha.q_proj.bias = torch.nn.Parameter(org_mha.in_proj_bias[:embed_dim].clone().detach())
        simple_mha.k_proj.weight = torch.nn.Parameter(org_mha.in_proj_weight[embed_dim:2 * embed_dim].clone().detach())
        simple_mha.k_proj.bias = torch.nn.Parameter(org_mha.in_proj_bias[embed_dim:2 * embed_dim].clone().detach())
        simple_mha.v_proj.weight = torch.nn.Parameter(
            org_mha.in_proj_weight[2 * embed_dim:3 * embed_dim].clone().detach())
        simple_mha.v_proj.bias = torch.nn.Parameter(org_mha.in_proj_bias[2 * embed_dim:3 * embed_dim].clone().detach())
        simple_mha.o_proj.weight = torch.nn.Parameter(org_mha.out_proj.weight.clone().detach())
        simple_mha.o_proj.bias = torch.nn.Parameter(org_mha.out_proj.bias.clone().detach())
    elif isinstance(org_mha, CausalSelfAttention):
        assert org_mha.n_embd == simple_mha.embed_dim
        assert org_mha.n_head == simple_mha.num_heads
        assert org_mha.c_attn.bias is not None
        assert org_mha.c_proj.bias is not None
        assert org_mha.bias is not None # causal self-attention mask
        embed_dim = org_mha.n_embd
        simple_mha.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(org_mha.bias.shape[-1], org_mha.bias.shape[-1])).view(
                1, 1, org_mha.bias.shape[-1], org_mha.bias.shape[-1]
            ),
        )
        simple_mha.q_proj.weight = torch.nn.Parameter(org_mha.c_attn.weight[:embed_dim].clone().detach())
        simple_mha.q_proj.bias = torch.nn.Parameter(org_mha.c_attn.bias[:embed_dim].clone().detach())
        simple_mha.k_proj.weight = torch.nn.Parameter(org_mha.c_attn.weight[embed_dim:2 * embed_dim].clone().detach())
        simple_mha.k_proj.bias = torch.nn.Parameter(org_mha.c_attn.bias[embed_dim:2 * embed_dim].clone().detach())
        simple_mha.v_proj.weight = torch.nn.Parameter(
            org_mha.c_attn.weight[2 * embed_dim:3 * embed_dim].clone().detach())
        simple_mha.v_proj.bias = torch.nn.Parameter(org_mha.c_attn.bias[2 * embed_dim:3 * embed_dim].clone().detach())
        simple_mha.o_proj.weight = torch.nn.Parameter(org_mha.c_proj.weight.clone().detach())
        simple_mha.o_proj.bias = torch.nn.Parameter(org_mha.c_proj.bias.clone().detach())
    else:
        raise NotImplementedError()


def simplify_mha(model):
    mhas_names = find_module_names(model, lambda _model, m: isinstance(m, (nn.MultiheadAttention, CausalSelfAttention)))
    for mha_name in mhas_names:
        org_mha = get_module_by_name(model, mha_name)
        if isinstance(org_mha, nn.MultiheadAttention):
            embed_dim = org_mha.embed_dim
            num_heads = org_mha.num_heads
            dropout = org_mha.dropout
        elif isinstance(org_mha, CausalSelfAttention):
            embed_dim = org_mha.n_embd
            num_heads = org_mha.n_head
            dropout = org_mha.dropout
        else:
            raise NotImplementedError()

        simple_mha = CustomMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout_p=dropout)
        transcribe_mha_params(org_mha, simple_mha)
        set_module_by_name(model, mha_name, simple_mha)


def create_attention_projection_filter_condition(projection_name: str = None):
    def filter_condition(model: nn.Module, m: nn.Module):
        m_name = get_module_name(model, m)
        parent_module = get_module_by_name(model, get_parent_module_name(m_name))
        if isinstance(m, nn.Linear) and isinstance(parent_module, CustomMultiheadAttention):
            if projection_name is not None and projection_name not in m_name:
                return False
            return True

    return filter_condition
