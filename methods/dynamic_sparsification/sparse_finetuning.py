import logging
from datetime import datetime
from typing import Dict, List

import torch
from omegaconf import OmegaConf
from torch import nn
from torchvision.ops import MLP as TorchvisionMLP
from transformers.activations import GELUActivation, PytorchGELUTanh
from transformers.models.bert.modeling_bert import BertIntermediate
from transformers.models.gemma.modeling_gemma import GemmaMLP

from architectures.gpt import MLP as GPTMLP
from architectures.moe.dsti import SimpleMLP, find_relu_activations
from architectures.vit import MLP
from common import INIT_NAME_MAP, get_default_args
from train import (
    TrainingContext,
    final_eval,
    in_training_eval,
    setup_accelerator,
    setup_data,
    setup_files_and_logging,
    setup_optimization,
    setup_state,
)
from utils import (
    Mixup,
    add_save_activations_hook,
    find_module_names,
    get_lrs,
    get_module_by_name,
    get_module_name,
    get_parent_module_name,
    load_model,
    save_state,
    set_module_by_name,
)


class EnforceSparsityTrainingContext(TrainingContext):
    sparsity_enforcement_mode: str = None
    modules_inputs: Dict = None
    modules_outputs: Dict = None
    model_hook_handles: List = None


def eligible_activation_filter(model: nn.Module, m: nn.Module):
    # TODO handle cases when functional variant is used instead
    m_name = get_module_name(model, m)
    parent_module = get_module_by_name(model, get_parent_module_name(m_name))
    if isinstance(m, (nn.ReLU, nn.GELU, GELUActivation, PytorchGELUTanh)) and isinstance(parent_module, (
            MLP, TorchvisionMLP, SimpleMLP, GPTMLP, BertIntermediate, GemmaMLP)):
        return True


def find_activations(model, apply_to):
    if apply_to == 'moe_eligible_only':
        acts_to_sparsify = find_module_names(model, eligible_activation_filter)
    elif apply_to == 'everywhere':
        acts_to_sparsify = find_module_names(model, lambda _, m: isinstance(m, (
        nn.ReLU, nn.GELU, GELUActivation, PytorchGELUTanh)))
    else:
        raise ValueError(f'Invalid apply_to argument value: {apply_to}')
    return acts_to_sparsify


def replace_with_relu(model, acts_to_replace):
    for act_m_name in acts_to_replace:
        logging.info(f'Replacing {act_m_name} with ReLU')
        set_module_by_name(model, act_m_name, nn.ReLU())
    return model


def setup_model(args, tc):
    assert args.model_class == 'enforce_sparsity'
    model, base_args, _ = load_model(args, args.base_on, args.exp_id)
    model = tc.accelerator.prepare(model)
    activations_to_sparsify = find_activations(model, **args.model_args)
    if 'relu' in args.dsti_enforce_mode:
        model = replace_with_relu(model, activations_to_sparsify)
        # TODO add "apply_to" argument to find_relu_activations
        activations_to_sparsify.extend(find_relu_activations(model))
    logging.info(f'Modules selected for activation sparsification: {activations_to_sparsify}')
    tc.modules_inputs, tc.modules_outputs, tc.model_hook_handles = \
        add_save_activations_hook(model, activations_to_sparsify)
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        init_fun(tc.model)
    tc.model = tc.accelerator.prepare(model)
    # make sure all parameters are being optimized
    tc.model.requires_grad_(True)


def square_hoyer(tensor, dim=-1, eps=1e-15):
    """As in http://proceedings.mlr.press/v119/kurtz20a/kurtz20a.pdf Section 2.1"""
    return torch.linalg.vector_norm(tensor, ord=1.0, dim=dim) ** 2 / \
        (torch.linalg.vector_norm(tensor, ord=2.0, dim=dim) ** 2 + eps)


def training_loop(args, tc):
    if tc.accelerator.is_main_process:
        model_saved = datetime.now()
    train_iter = iter(tc.train_loader)
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    if args.mixup_alpha is not None or args.cutmix_alpha is not None:
        mixup_mode = 'batch' if args.mixup_mode is None else args.mixup_mode
        mixup_smoothing = 0.1 if args.mixup_smoothing is None else args.mixup_smoothing
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, mode=mixup_mode,
            label_smoothing=mixup_smoothing, num_classes=unwrapped_model.number_of_classes)
    else:
        mixup_fn = None
    while tc.state.current_batch < tc.last_batch:
        # save model conditionally
        if tc.accelerator.is_main_process:
            now = datetime.now()
            if (now - model_saved).total_seconds() > 60 * args.save_every:
                save_state(tc.accelerator, tc.state_path)
                model_saved = datetime.now()
        # model evaluation
        in_training_eval(args, tc)
        tc.model.train()
        tc.optimizer.zero_grad(set_to_none=True)
        # Account for gradient accumulation
        cumulative_task_loss = 0
        cumulative_sparsity_loss_ffn = 0
        cumulative_sparsity_loss_attn = 0
        cumulativte_total_loss = 0
        for _ in range(args.gradient_accumulation_steps):
            # batch preparation
            try:
                X, y = next(train_iter)
            except StopIteration:
                train_iter = iter(tc.train_loader)
                X, y = next(train_iter)
            if mixup_fn is not None:
                X, y = mixup_fn(X, y)
            # forward
            y_pred = tc.model(X)
            if len(y_pred.shape) == 3:
                # For CE loss on sequences
                y_pred = y_pred.transpose(1, 2)
            # get activations / pre-activations and compute sparsity loss
            sparsity_loss_ffn = 0.0
            sparse_activations_sum_ffn = 0
            total_activations_sum_ffn = 0
            total_num_outputs_ffn = 0
            sparsity_loss_attn = 0.0
            sparse_activations_sum_attn = 0
            total_activations_sum_attn = 0
            total_num_outputs_attn = 0
            # warning! inputs (e.g. pre-activations for gelu) are tuples, outputs (e.g. relu activations) are tensors
            if args.dsti_enforce_mode == 'gelu_preactivations_hoyer_clamped':
                for module_name, preacts in tc.modules_inputs.items():
                    # get first input tensor from the input tuple
                    assert isinstance(preacts, tuple)
                    preacts = preacts[0]
                    clamped_preactivations = preacts.clamp(
                        min=args.dsti_clamp_displacement) - args.dsti_clamp_displacement
                    sparsity_loss = square_hoyer(clamped_preactivations, dim=-1).mean()
                    sparse_activations_sum = (clamped_preactivations <= args.dsti_clamp_displacement).sum().item()
                    total_activations_sum = preacts.numel()
                    if 'q_proj' in module_name or 'k_proj' in module_name or 'v_proj' in module_name or 'o_proj' in module_name:
                        sparsity_loss_attn += sparsity_loss
                        sparse_activations_sum_attn += sparse_activations_sum
                        total_activations_sum_attn += total_activations_sum
                        total_num_outputs_attn += 1
                    else:
                        sparsity_loss_ffn += sparsity_loss
                        sparse_activations_sum_ffn += sparse_activations_sum
                        total_activations_sum_ffn += total_activations_sum
                        total_num_outputs_ffn += 1
            elif args.dsti_enforce_mode == 'gelu_preactivations_clamped':
                for module_name, preacts in tc.modules_inputs.items():
                    # get first input tensor from the input tuple
                    assert isinstance(preacts, tuple)
                    preacts = preacts[0]
                    clamped_preactivations = preacts.clamp(
                        min=args.dsti_clamp_displacement) - args.dsti_clamp_displacement
                    sparsity_loss = clamped_preactivations.sum(dim=-1).mean()
                    sparse_activations_sum = (clamped_preactivations <= args.dsti_clamp_displacement).sum().item()
                    total_activations_sum = preacts.numel()
                    if 'q_proj' in module_name or 'k_proj' in module_name or 'v_proj' in module_name or 'o_proj' in module_name:
                        sparsity_loss_attn += sparsity_loss
                        sparse_activations_sum_attn += sparse_activations_sum
                        total_activations_sum_attn += total_activations_sum
                        total_num_outputs_attn += 1
                    else:
                        sparsity_loss_ffn += sparsity_loss
                        sparse_activations_sum_ffn += sparse_activations_sum
                        total_activations_sum_ffn += total_activations_sum
                        total_num_outputs_ffn += 1
            elif args.dsti_enforce_mode == 'relu_hoyer':
                for module_name, acts in tc.modules_outputs.items():
                    assert isinstance(acts, torch.Tensor)
                    sparsity_loss = square_hoyer(acts, dim=-1).mean()
                    sparse_activations_sum = (acts <= 0).sum().item()
                    total_activations_sum = acts.numel()
                    if 'q_proj' in module_name or 'k_proj' in module_name or 'v_proj' in module_name or 'o_proj' in module_name:
                        sparsity_loss_attn += sparsity_loss
                        sparse_activations_sum_attn += sparse_activations_sum
                        total_activations_sum_attn += total_activations_sum
                        total_num_outputs_attn += 1
                    else:
                        sparsity_loss_ffn += sparsity_loss
                        sparse_activations_sum_ffn += sparse_activations_sum
                        total_activations_sum_ffn += total_activations_sum
                        total_num_outputs_ffn += 1
            elif args.dsti_enforce_mode == 'relu_l1':
                for module_name, acts in tc.modules_outputs.items():
                    assert isinstance(acts, torch.Tensor)
                    sparsity_loss = acts.sum(dim=-1).mean()
                    sparse_activations_sum = (acts <= 0).sum().item()
                    total_activations_sum = acts.numel()
                    if 'q_proj' in module_name or 'k_proj' in module_name or 'v_proj' in module_name or 'o_proj' in module_name:
                        sparsity_loss_attn += sparsity_loss
                        sparse_activations_sum_attn += sparse_activations_sum
                        total_activations_sum_attn += total_activations_sum
                        total_num_outputs_attn += 1
                    else:
                        sparsity_loss_ffn += sparsity_loss
                        sparse_activations_sum_ffn += sparse_activations_sum
                        total_activations_sum_ffn += total_activations_sum
                        total_num_outputs_ffn += 1
            else:
                raise ValueError(f'{args.enforce_mode=}')
            exact_sparsity = 0
            approx_sparsity_10 = 0
            approx_sparsity_100 = 0
            approx_sparsity_1000 = 0
            approx_sparsity_10000 = 0
            for module_name, acts in tc.modules_outputs.items():
                acts = acts.abs()
                acts = acts.view(-1, acts.size(-1))
                max_acts = acts.max(dim=-1).values.unsqueeze(1)
                exact_sparsity += (acts == 0.).sum().item()
                approx_sparsity_10 += (acts < max_acts / 10).sum().item()
                approx_sparsity_100 += (acts < max_acts / 100).sum().item()
                approx_sparsity_1000 += (acts < max_acts / 1000).sum().item()
                approx_sparsity_10000 += (acts < max_acts / 10000).sum().item()
            exact_sparsity /= (total_activations_sum_attn + total_activations_sum_ffn)
            approx_sparsity_10 /= (total_activations_sum_attn + total_activations_sum_ffn)
            approx_sparsity_100 /= (total_activations_sum_attn + total_activations_sum_ffn)
            approx_sparsity_1000 /= (total_activations_sum_attn + total_activations_sum_ffn)
            approx_sparsity_10000 /= (total_activations_sum_attn + total_activations_sum_ffn)
            if total_num_outputs_attn > 0:
                sparsity_loss_attn /= total_num_outputs_attn
            sparsity_loss_ffn /= total_num_outputs_ffn
            # task loss
            task_loss = tc.criterion(y_pred, y)
            if args.dsti_enforce_schedule == 'linear':
                # main
                dsti_enforce_weight = tc.state.current_batch / tc.last_batch * args.dsti_enforce_weight
                if tc.accelerator.is_main_process and tc.state.current_batch in tc.eval_batch_list:
                    tc.writer.add_scalar(f'Train/Enforce weight FFN', dsti_enforce_weight,
                                         global_step=tc.state.current_batch)
            elif args.dsti_enforce_schedule == 'linear_warmup':
                max_weight_reach_batch = int(tc.last_batch / 2)
                dsti_enforce_weight = min(tc.state.current_batch / max_weight_reach_batch,
                                          1.0) * args.dsti_enforce_weight
                if tc.accelerator.is_main_process and tc.state.current_batch in tc.eval_batch_list:
                    tc.writer.add_scalar(f'Train/Enforce weight', dsti_enforce_weight,
                                         global_step=tc.state.current_batch)
                #
            elif args.dsti_enforce_schedule is None:
                dsti_enforce_weight = args.dsti_enforce_weight
            else:
                raise NotImplementedError()
            if args.gradient_accumulation_steps > 1:
                task_loss = task_loss / args.gradient_accumulation_steps
                sparsity_loss_ffn = sparsity_loss_ffn / args.gradient_accumulation_steps
                sparsity_loss_attn = sparsity_loss_attn / args.gradient_accumulation_steps
            # loss computation
            loss = (task_loss +
                    dsti_enforce_weight * sparsity_loss_ffn +
                    dsti_enforce_weight * sparsity_loss_attn)
            tc.accelerator.backward(loss)
            cumulativte_total_loss += loss.item()
            cumulative_task_loss += task_loss.item()
            cumulative_sparsity_loss_ffn += float(sparsity_loss_ffn)
            cumulative_sparsity_loss_attn += float(sparsity_loss_attn)
        # gradient computation and training step
        if args.clip_grad_norm is not None:
            total_norm = tc.accelerator.clip_grad_norm_(tc.model.parameters(), args.clip_grad_norm)
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar('Train/Gradient norm', total_norm.item(), global_step=tc.state.current_batch)
        tc.optimizer.step()
        if tc.scheduler is not None:
            # log LRs
            if tc.accelerator.is_main_process:
                for i, lr in enumerate(get_lrs(tc.optimizer)):
                    tc.writer.add_scalar(f'Train/Group {i} LR', lr, global_step=tc.state.current_batch)
            if args.scheduler_class == 'reduce_on_plateau':
                tc.scheduler.step(loss)
            else:
                tc.scheduler.step()
        # bookkeeping
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Train/Task loss', cumulative_task_loss, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Train/Sparsity loss FFN', cumulative_sparsity_loss_ffn,
                                 global_step=tc.state.current_batch)
            tc.writer.add_scalar('Train/Sparsity loss Attn', cumulative_sparsity_loss_attn,
                                 global_step=tc.state.current_batch)
            tc.writer.add_scalar('Train/Loss', cumulativte_total_loss, global_step=tc.state.current_batch)

            sparsity_ffn = sparse_activations_sum_ffn / total_activations_sum_ffn
            tc.writer.add_scalar('Train/Sparsity FFN', sparsity_ffn, global_step=tc.state.current_batch)
            if total_num_outputs_attn > 0.:
                sparsity_attn = sparse_activations_sum_attn / total_activations_sum_attn
                tc.writer.add_scalar('Train/Sparsity Attn', sparsity_attn, global_step=tc.state.current_batch)
                sparsity_total = (sparse_activations_sum_ffn + sparse_activations_sum_attn) / (
                        total_activations_sum_ffn + total_activations_sum_attn)
                tc.writer.add_scalar('Train/Sparsity total', sparsity_total, global_step=tc.state.current_batch)

            tc.writer.add_scalar('Activation sparsity train/Exact', exact_sparsity, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Activation sparsity train/Approx 10', approx_sparsity_10,
                                 global_step=tc.state.current_batch)
            tc.writer.add_scalar('Activation sparsity train/Approx 100', approx_sparsity_100,
                                 global_step=tc.state.current_batch)
            tc.writer.add_scalar('Activation sparsity train/Approx 1000', approx_sparsity_1000,
                                 global_step=tc.state.current_batch)
            tc.writer.add_scalar('Activation sparsity train/Approx 10000', approx_sparsity_10000,
                                 global_step=tc.state.current_batch)
        tc.state.current_batch += 1


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
    tc = EnforceSparsityTrainingContext()
    tc.sparsity_enforcement_mode = args.dsti_enforce_mode
    tc.sparsity_enforcement_displacement = args.dsti_clamp_displacement
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_data(args, tc)
    setup_model(args, tc)
    setup_optimization(args, tc)
    setup_state(tc)
    training_loop(args, tc)
    final_eval(args, tc)


def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()
