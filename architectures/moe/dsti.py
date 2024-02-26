import logging
from typing import Callable, List, Optional, Union

import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer as TorchvisionViT
from torchvision.ops import MLP as TorchvisionMLP
from transformers import BertLayer, BertPreTrainedModel
from transformers.activations import GELUActivation
from transformers.models.bert.modeling_bert import BertIntermediate, BertSelfAttention, BertSelfOutput

from architectures.custom import CustomMultiheadAttention, create_attention_projection_filter_condition
from architectures.gpt import MLP as GPTMLP
from architectures.vit import MLP, VisionTransformer
from utils import find_module_names, get_module_by_name, get_module_name, get_parent_module_name, set_module_by_name


class SubstitutionMLP(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SimpleMLP(torch.nn.Sequential, SubstitutionMLP):
    # Almost the same as MLP from torchvison's ViT
    def __init__(
        self,
        in_size: int,
        hidden_sizes: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation: str = "gelu",
        bias: bool = True,
        dropout: float = 0.0,
    ):
        from common import ACTIVATION_NAME_MAP

        layers = []
        in_dim = in_size
        for hidden_dim in hidden_sizes[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(ACTIVATION_NAME_MAP[activation]())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, hidden_sizes[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout))
        torch.nn.Sequential.__init__(self, *layers)


class ResidualMLP(torch.nn.Sequential, SubstitutionMLP):
    def __init__(
        self,
        in_size: int,
        hidden_sizes: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation: str = "gelu",
        bias: bool = True,
        dropout: float = 0.0,
    ):
        SubstitutionMLP.__init__(self)
        from common import ACTIVATION_NAME_MAP

        layers = []
        in_dim = in_size
        for hidden_dim in hidden_sizes[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(ACTIVATION_NAME_MAP[activation]())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, hidden_sizes[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout))
        torch.nn.Sequential.__init__(self, *layers)

    def forward(self, x):
        saved_x = x
        for layer in self:
            x = layer(x)
        x = x + saved_x
        return x


MLP_NAME_MAP = {"simple": SimpleMLP, "residual": ResidualMLP}


def replace_mha_projections(
    model: nn.Module,
    flops_factor: float,
    activation: str = "gelu",
    bias: bool = None,
    dropout: Union[float, str] = 0.0,
    q_proj: bool = True,
    k_proj: bool = True,
    v_proj: bool = True,
    o_proj: bool = True,
    mlp_type: str = "simple",
):
    model.eval()
    if isinstance(model, (VisionTransformer, TorchvisionViT)):
        hidden_dim = model.hidden_dim
    elif isinstance(model, BertPreTrainedModel):
        hidden_dim = model.config.hidden_size
    attention_dim = round(flops_factor * (hidden_dim**2 + hidden_dim) / (2 * hidden_dim + 1))
    if dropout == "same":
        if isinstance(model, (VisionTransformer, TorchvisionViT)):
            dropout = model.attention_dropout
        elif isinstance(model, BertPreTrainedModel):
            dropout = model.config.attention_probs_dropout_prob
    logging.info(
        f"Determined attention_dim: {attention_dim}" f" (activation: {activation}, bias:{bias}, dropout: {dropout})"
    )
    # avoid deepcopying the model, because it's crasing jit.trace
    modules_to_replace = []
    if o_proj:
        if isinstance(model, (VisionTransformer, TorchvisionViT)):
            modules_to_replace += find_module_names(model, create_attention_projection_filter_condition("o_proj"))
        elif isinstance(model, BertPreTrainedModel):
            modules_to_replace += find_module_names(model, create_attention_projection_filter_condition("dense", attention_class=BertSelfOutput))
    if q_proj:
        if isinstance(model, (VisionTransformer, TorchvisionViT)):
            modules_to_replace += find_module_names(model, create_attention_projection_filter_condition("q_proj"))
        elif isinstance(model, BertPreTrainedModel):
            modules_to_replace += find_module_names(model, create_attention_projection_filter_condition("query", attention_class=BertSelfAttention))
    if k_proj:
        if isinstance(model, (VisionTransformer, TorchvisionViT)):
            modules_to_replace += find_module_names(model, create_attention_projection_filter_condition("k_proj"))
        elif isinstance(model, BertPreTrainedModel):
            modules_to_replace += find_module_names(model, create_attention_projection_filter_condition("key", attention_class=BertSelfAttention))
    if v_proj:
        if isinstance(model, (VisionTransformer, TorchvisionViT)):
            modules_to_replace += find_module_names(model, create_attention_projection_filter_condition("v_proj"))
        elif isinstance(model, BertPreTrainedModel):
            modules_to_replace += find_module_names(model, create_attention_projection_filter_condition("value", attention_class=BertSelfAttention))
    # use hooks to get an example input
    assert len(modules_to_replace) > 0, f"{modules_to_replace=}"
    # replace the selected layers
    for name in modules_to_replace:
        original_module = get_module_by_name(model, name)
        bias = bias if bias is not None else original_module.bias is not None
        replacement = MLP_NAME_MAP[mlp_type](
            hidden_dim, [attention_dim, hidden_dim], activation=activation, bias=bias, dropout=dropout
        )
        set_module_by_name(model, name, replacement)
        logging.info(f"Substituting {name}")
    return model, modules_to_replace



def dsti_mlp_filter_condition(_model: nn.Module, m: nn.Module):
    if isinstance(m, (TorchvisionMLP, MLP, SubstitutionMLP, GPTMLP, BertLayer)):
        return True


def mha_activation_filter(model: nn.Module, m: nn.Module):
    m_name = get_module_name(model, m)
    parent_m_name = get_parent_module_name(m_name)
    grandparent_m_name = get_parent_module_name(parent_m_name)
    parent_module = get_module_by_name(model, parent_m_name)
    grandparent_module = get_module_by_name(model, grandparent_m_name)
    if (
        isinstance(m, (nn.ReLU, nn.GELU, GELUActivation))
        and isinstance(parent_module, SubstitutionMLP)
        and isinstance(grandparent_module, (CustomMultiheadAttention, BertSelfAttention, BertSelfOutput))
    ):
        return True


def eligible_activation_filter(model: nn.Module, m: nn.Module):
    # TODO handle cases when functional variant is used instead
    m_name = get_module_name(model, m)
    parent_module = get_module_by_name(model, get_parent_module_name(m_name))
    if isinstance(m, (nn.ReLU, nn.GELU, GELUActivation)) and isinstance(
        parent_module, (MLP, TorchvisionMLP, SubstitutionMLP, GPTMLP, BertIntermediate)
    ):
        return True


def find_activations(model, apply_to):
    if apply_to == "moe_eligible_only":
        acts_to_sparsify = find_module_names(model, eligible_activation_filter)
    elif apply_to == "everywhere":
        acts_to_sparsify = find_module_names(model, lambda _, m: isinstance(m, (nn.ReLU, nn.GELU, GELUActivation)))
    elif apply_to == "mha_projections":
        acts_to_sparsify = find_module_names(model, mha_activation_filter)
    else:
        raise ValueError(f"Invalid apply_to argument value: {apply_to}")
    return acts_to_sparsify


def replace_with_relu(model, acts_to_replace):
    for act_m_name in acts_to_replace:
        logging.info(f"Replacing {act_m_name} with ReLU")
        set_module_by_name(model, act_m_name, nn.ReLU())
    return model
