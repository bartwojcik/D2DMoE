import logging
from copy import deepcopy

from omegaconf import OmegaConf
from torch import nn
from torchvision.ops import MLP as TorchvisionMLP
from transformers.activations import PytorchGELUTanh
from transformers.models.gemma.modeling_gemma import GemmaMLP

from architectures.gpt import MLP as GPTMLP
from architectures.vit import MLP
from common import get_default_args, INIT_NAME_MAP
from train import TrainingContext, setup_accelerator, setup_files_and_logging, setup_data, setup_optimization, \
    setup_state, training_loop, final_eval
from utils import load_model, find_module_names, get_module_name, get_module_by_name, get_parent_module_name, \
    set_module_by_name


def inside_ffn_filter(model: nn.Module, m: nn.Module):
    m_name = get_module_name(model, m)
    parent_module = get_module_by_name(model, get_parent_module_name(m_name))
    # LLaMa uses SiLU
    if isinstance(m, (nn.GELU, PytorchGELUTanh)) and isinstance(parent_module, (MLP, TorchvisionMLP, GPTMLP, GemmaMLP)):
        return True


def replace_with_relu(original_model, mode):
    model = deepcopy(original_model)
    if mode == 'ffns_only':
        acts_to_replace = find_module_names(model, inside_ffn_filter)
    elif mode == 'everywhere':
        acts_to_replace = find_module_names(model, lambda _, m: isinstance(m, (nn.GELU, PytorchGELUTanh)))
    else:
        raise ValueError(f'Invalid relufication mode: {mode}')
    for act_m_name in acts_to_replace:
        logging.info(f'Replacing {act_m_name} with ReLU')
        set_module_by_name(model, act_m_name, nn.ReLU())
    return model


def setup_model(args, tc):
    assert args.model_class == 'relufication'
    model, _, _ = load_model(args, args.base_on, args.exp_id)
    model = replace_with_relu(model, **args.model_args)
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        init_fun(model)
    tc.model = tc.accelerator.prepare(model)
    tc.model.train()


def train(args):
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info('Configured logging')
    tc = TrainingContext()
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_model(args, tc)
    setup_data(args, tc)
    setup_optimization(args, tc)
    setup_state(tc)
    training_loop(args, tc)
    final_eval(args, tc)


def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()
