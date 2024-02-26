import sys

import torch
from transformers import BertPreTrainedModel

from architectures.moe.moe_models import ffn_filter_condition_bert

if 'conda' in sys.path[-1] and 'site-packages' in sys.path[-1] and '.local' in sys.path[-2] and 'site-packages' in \
        sys.path[-2]:
    print("Fixing sys.path")
    sys.path[-2], sys.path[-1] = sys.path[-1], sys.path[-2]

import logging
from typing import List

from omegaconf import OmegaConf
from torch import nn

from architectures.moe.moefication import replace_with_moes, split_original_parameters
from architectures.vit import ffn_filter_condition as ffn_filter_condition_vit, VisionTransformer
from torchvision.models.vision_transformer import VisionTransformer as TorchvisionViT
from architectures.gpt import ffn_filter_condition as ffn_filter_condition_gpt, GPT
from common import get_default_args, INIT_NAME_MAP
from train import TrainingContext, setup_accelerator, setup_data, setup_optimization, setup_files_and_logging, \
    setup_state, final_eval
from utils import load_model


class ParamSplitContext(TrainingContext):
    orig_model: nn.Module = None
    replaced_modules_list: List[str] = None


def setup_model(args, tc):
    assert args.model_class == 'moefication'
    base_model, base_args, _ = load_model(args, args.base_on, args.exp_id)
    tc.orig_model = base_model
    if isinstance(base_model, (VisionTransformer, TorchvisionViT)):
        ffn_filter_condition = ffn_filter_condition_vit
    elif isinstance(base_model, GPT):
        ffn_filter_condition = ffn_filter_condition_gpt
    elif isinstance(base_model, BertPreTrainedModel):
        ffn_filter_condition = ffn_filter_condition_bert
    else:
        raise NotImplementedError(f'Unknown model type: {type(base_model)}')
    tc.model, tc.replaced_modules_list = replace_with_moes(base_model, **args.model_args,
                                                           module_filter_contition=ffn_filter_condition)
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        init_fun(tc.model)
    tc.model = tc.accelerator.prepare(tc.model)


def param_split(tc):
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    if tc.state.current_batch == 0:
        split_original_parameters(tc.orig_model, unwrapped_model, tc.replaced_modules_list)
        tc.state.current_batch = max(tc.last_batch + 1, 1)
    del tc.orig_model


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
    tc = ParamSplitContext()
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_model(args, tc)
    setup_data(args, tc)
    setup_optimization(args, tc)
    setup_state(tc)
    param_split(tc)
    final_eval(args, tc)


def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()
