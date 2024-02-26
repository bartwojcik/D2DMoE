import logging
from datetime import datetime
from typing import Dict, List

import torch
from omegaconf import OmegaConf

from architectures.moe.dsti import find_activations, replace_with_relu
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
from utils import Mixup, add_save_activations_hook, get_lrs, load_model, save_state


class EnforceSparsityTrainingContext(TrainingContext):
    sparsity_enforcement_mode: str = None
    modules_inputs: Dict = None
    modules_outputs: Dict = None
    model_hook_handles: List = None
    mha_modules_names: List = None


def setup_model(args, tc):
    logging.info('Setting up model')
    assert args.model_class == 'enforce_sparsity'
    model, base_args, _ = load_model(args, args.base_on, args.exp_id)
    model = tc.accelerator.prepare(model)
    activations_to_sparsify = find_activations(model, **args.model_args)
    tc.mha_modules_names = find_activations(model, apply_to='mha_projections')
    if 'relu' in args.dsti_enforce_mode:
        model = replace_with_relu(model, activations_to_sparsify)
    logging.info(f'Modules selected for activation sparsification: {activations_to_sparsify}')
    tc.modules_inputs, tc.modules_outputs, tc.model_hook_handles = \
        add_save_activations_hook(model, activations_to_sparsify)
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        init_fun(tc.model)
    tc.model = tc.accelerator.prepare(model)
    # make sure all parameters are being optimized
    tc.model.requires_grad_(True)
    tc.sparsity_enforcement_mode = args.dsti_enforce_mode
    tc.sparsity_enforcement_displacement = args.dsti_clamp_displacement
    logging.info('Model set up')


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
    while tc.state.current_batch <= tc.last_batch:
        # save model conditionally
        if tc.accelerator.is_main_process:
            now = datetime.now()
            if (now - model_saved).total_seconds() > 60 * args.save_every:
                save_state(tc.accelerator, tc.state_path)
                model_saved = datetime.now()
        # model evaluation
        in_training_eval(args, tc)

        tc.model.train()

        # Account for gradient accumulation
        cumulative_task_loss = 0.0
        cumulative_sparsity_loss_ffn = 0.0
        cumulative_sparsity_loss_mha = 0.0
        cumulativte_loss = 0.0

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

            sparsity_loss_mha = 0.0
            sparse_activations_sum_mha = 0
            total_activations_sum_mha = 0

            if args.dsti_enforce_schedule == 'linear':
                dsti_enforce_weight_ffn = tc.state.current_batch / tc.last_batch * args.dsti_enforce_weight
                if tc.accelerator.is_main_process and tc.state.current_batch in tc.eval_batch_list:
                    tc.writer.add_scalar('Train/Enforce weight FFN', dsti_enforce_weight_ffn, global_step=tc.state.current_batch)
                if args.dsti_enforce_weight_mha is None:
                    dsti_enforce_weight_mha = dsti_enforce_weight_ffn
                else:
                    dsti_enforce_weight_mha = tc.state.current_batch / tc.last_batch * args.dsti_enforce_weight_mha
                    if tc.accelerator.is_main_process and tc.state.current_batch in tc.eval_batch_list:
                        tc.writer.add_scalar('Train/Enforce weight MHA', dsti_enforce_weight_mha, global_step=tc.state.current_batch)
            
            elif args.dsti_enforce_schedule is None:
                dsti_enforce_weight_ffn = args.dsti_enforce_weight
                if args.dsti_enforce_weight_mha is None:
                    dsti_enforce_weight_mha = dsti_enforce_weight_ffn
                else:
                    dsti_enforce_weight_mha = args.dsti_enforce_weight_mha

            # warning! inputs (e.g. pre-activations for gelu) are tuples, outputs (e.g. relu activations) are tensors
            if args.dsti_enforce_mode == 'gelu_preactivations_hoyer_clamped':
                for module_name, preacts in tc.modules_inputs.items():
                    # get first input tensor from the input tuple
                    assert isinstance(preacts, tuple)
                    preacts = preacts[0]
                    clamped_preactivations = preacts.clamp(min=args.dsti_clamp_displacement) - args.dsti_clamp_displacement
                    sparsity_loss = square_hoyer(clamped_preactivations, dim=-1).mean()
                    sparse_activations_sum = (clamped_preactivations <= args.dsti_clamp_displacement).sum().item()
                    total_activations_sum = preacts.numel()
                    if module_name in tc.mha_modules_names:
                        sparsity_loss_mha += sparsity_loss
                        sparse_activations_sum_mha += sparse_activations_sum
                        total_activations_sum_mha += total_activations_sum
                    else:
                        sparsity_loss_ffn += sparsity_loss
                        sparse_activations_sum_ffn += sparse_activations_sum
                        total_activations_sum_ffn += total_activations_sum
                sparsity_loss_mha /= len(tc.mha_modules_names)
                sparsity_loss_ffn /= (len(tc.modules_inputs) - len(tc.mha_modules_names))
            elif args.dsti_enforce_mode == 'gelu_preactivations_clamped':
                for module_name, preacts in tc.modules_inputs.items():
                    # get first input tensor from the input tuple
                    assert isinstance(preacts, tuple)
                    preacts = preacts[0]
                    clamped_preactivations = preacts.clamp(min=args.dsti_clamp_displacement) - args.dsti_clamp_displacement
                    sparsity_loss = clamped_preactivations.sum(dim=-1).mean()
                    sparse_activations_sum = (clamped_preactivations <= args.dsti_clamp_displacement).sum().item()
                    total_activations_sum = preacts.numel()
                    if module_name in tc.mha_modules_names:
                        sparsity_loss_mha += sparsity_loss
                        sparse_activations_sum_mha += sparse_activations_sum
                        total_activations_sum_mha += total_activations_sum
                    else:
                        sparsity_loss_ffn += sparsity_loss
                        sparse_activations_sum_ffn += sparse_activations_sum
                        total_activations_sum_ffn += total_activations_sum
                sparsity_loss_mha /= len(tc.mha_modules_names)
                sparsity_loss_ffn /= (len(tc.modules_inputs) - len(tc.mha_modules_names))
            elif args.dsti_enforce_mode == 'relu_hoyer':
                for module_name, acts in tc.modules_outputs.items():
                    assert isinstance(acts, torch.Tensor)
                    sparsity_loss = square_hoyer(acts, dim=-1).mean()
                    sparse_activations_sum = (acts <= 0).sum().item()
                    total_activations_sum = acts.numel()
                    if module_name in tc.mha_modules_names:
                        sparsity_loss_mha += sparsity_loss
                        sparse_activations_sum_mha += sparse_activations_sum
                        total_activations_sum_mha += total_activations_sum
                    else:
                        sparsity_loss_ffn += sparsity_loss
                        sparse_activations_sum_ffn += sparse_activations_sum
                        total_activations_sum_ffn += total_activations_sum
                sparsity_loss_mha /= len(tc.mha_modules_names)
                sparsity_loss_ffn /= (len(tc.modules_inputs) - len(tc.mha_modules_names))
            elif args.dsti_enforce_mode == 'relu_l1':
                for module_name, acts in tc.modules_outputs.items():
                    assert isinstance(acts, torch.Tensor)
                    sparsity_loss = acts.sum(dim=-1).mean()
                    sparse_activations_sum = (acts <= 0).sum().item()
                    total_activations_sum = acts.numel()
                    if module_name in tc.mha_modules_names:
                        sparsity_loss_mha += sparsity_loss
                        sparse_activations_sum_mha += sparse_activations_sum
                        total_activations_sum_mha += total_activations_sum
                    else:
                        sparsity_loss_ffn += sparsity_loss
                        sparse_activations_sum_ffn += sparse_activations_sum
                        total_activations_sum_ffn += total_activations_sum
                sparsity_loss_mha /= len(tc.mha_modules_names)
                sparsity_loss_ffn /= (len(tc.modules_inputs) - len(tc.mha_modules_names))
            else:
                raise ValueError(f'{args.enforce_mode=}')
            # task loss
            task_loss = tc.criterion(y_pred, y)

            # loss computation
            loss = task_loss + dsti_enforce_weight_ffn * sparsity_loss_ffn + dsti_enforce_weight_mha * sparsity_loss_mha

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tc.accelerator.backward(loss)

            cumulativte_loss += loss.item()
            cumulative_task_loss += task_loss.item()
            cumulative_sparsity_loss_ffn += sparsity_loss_ffn.item()
            cumulative_sparsity_loss_mha += sparsity_loss_mha.item()

        # gradient computation and training step
        # TODO not sure if we should not do clipping inside gradient accumulation loop
        if args.clip_grad_norm is not None:
            total_norm = tc.accelerator.clip_grad_norm_(tc.model.parameters(), args.clip_grad_norm)
            if tc.accelerator.is_main_process and total_norm is not None:  # TODO check if total_norm=None indicates a bug??
                tc.writer.add_scalar('Train/Gradient norm', total_norm.item(), global_step=tc.state.current_batch)
        tc.optimizer.step()
        tc.optimizer.zero_grad(set_to_none=True)

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
            tc.writer.add_scalar('Train/Sparsity loss FFN', cumulative_sparsity_loss_ffn, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Train/Sparsity loss MHA', cumulative_sparsity_loss_mha, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Train/Loss', cumulativte_loss, global_step=tc.state.current_batch)
            sparsity = (sparse_activations_sum_ffn + sparse_activations_sum_mha) / (
                        total_activations_sum_ffn + total_activations_sum_mha)
            sparsity_ffn = sparse_activations_sum_ffn / total_activations_sum_ffn
            sparsity_mha = sparse_activations_sum_mha / total_activations_sum_mha
            tc.writer.add_scalar('Train/Sparsity', sparsity, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Train/Sparsity FFN', sparsity_ffn, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Train/Sparsity MHA', sparsity_mha, global_step=tc.state.current_batch)


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