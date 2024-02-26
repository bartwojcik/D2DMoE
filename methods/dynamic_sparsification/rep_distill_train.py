import logging
from datetime import datetime
from typing import Dict, List, Type

import torch
from omegaconf import OmegaConf
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer as TorchvisionViT

from architectures.custom import simplify_mha
from architectures.moe.dsti import SubstitutionMLP, find_activations, replace_mha_projections, replace_with_relu
from architectures.vit import VisionTransformer
from common import INIT_NAME_MAP, LOSS_NAME_MAP, get_default_args
from methods.dynamic_sparsification.sparse_finetuning import square_hoyer
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
    configure_logging,
    get_lrs,
    get_module_by_name,
    load_model,
    save_state,
)


class ReplaceModulesTrainingContext(TrainingContext):
    base_model: torch.nn.Module = None
    replaced_module_names: List[str] = None
    replacement_modules: Dict[str, nn.Module] = None
    distill_criterion_type: Type = None
    base_modules_inputs: Dict = None
    base_modules_outputs: Dict = None
    base_model_hook_handles: List = None
    sparsity_modules_inputs: Dict = None
    sparsity_modules_outputs: Dict = None
    sparsity_model_hook_handles: List = None


def setup_model(args, tc):
    assert args.model_class == 'mha_rep_distill'
    base_model, base_args, _ = load_model(args, args.base_on, args.exp_id)
    model, base_args, _ = load_model(args, args.base_on, args.exp_id)
    if isinstance(base_model, (VisionTransformer, TorchvisionViT)):
        simplify_mha(base_model)
    model_args = args.model_args
    model, tc.replaced_module_names = replace_mha_projections(model, **model_args)
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        init_fun(model)
    activations_to_sparsify = find_activations(model, 'mha_projections')
    if 'relu' in args.dsti_enforce_mode:
        model = replace_with_relu(model, activations_to_sparsify)
    logging.info(f'Modules selected for activation sparsification: {activations_to_sparsify}')
    tc.sparsity_modules_inputs, tc.sparsity_modules_outputs, tc.sparsity_model_hook_handles = \
        add_save_activations_hook(model, activations_to_sparsify)
    tc.replacement_modules = {name: get_module_by_name(model, name) for name in tc.replaced_module_names}
    tc.base_model = tc.accelerator.prepare(base_model)
    tc.model = tc.accelerator.prepare(model)


def set_for_distillation_iteration(tc):
    tc.base_model.eval()
    for m in tc.model.modules():
        if isinstance(m, SubstitutionMLP):
            m.train()


def setup_for_training(args, tc):
    tc.distill_criterion_type = LOSS_NAME_MAP[args.dsti_distill_loss_type]
    tc.base_model.eval()
    unwrapped_base_model = tc.accelerator.unwrap_model(tc.base_model)
    tc.base_modules_inputs, tc.base_modules_outputs, tc.base_model_hook_handles = \
        add_save_activations_hook(unwrapped_base_model, tc.replaced_module_names)
    tc.model.eval()
    tc.model.requires_grad_(False)
    for m in tc.model.modules():
        if isinstance(m, SubstitutionMLP):
            m.train()
            m.requires_grad_(True)


# def test_representation_distillation(tc,
#                                      data_loader: torch.utils.data.DataLoader,
#                                      batches: int = 0):
#     criterion = tc.distill_criterion_type(reduction='none')
#     set_for_eval(tc)
#     with torch.no_grad():
#         losses = defaultdict(list)
#         for batch, (X, _) in enumerate(data_loader):
#             tc.base_model(X)
#             # iterate over each layer, calculate loss for each one
#             for module_name in tc.replaced_module_names:
#                 module = tc.replacement_modules[module_name]
#                 original_input = tc.base_modules_inputs[module_name][0].detach()
#                 original_output = tc.base_modules_outputs[module_name].detach()
#                 output = module(original_input)
#                 output, original_output = tc.accelerator.gather_for_metrics((output, original_output))
#                 batch_losses = criterion(output, original_output)
#                 losses[module_name].append(batch_losses.detach().mean(dim=(1, 2)))
#                 if batch >= batches > 0:
#                     break
#         for k in losses.keys():
#             losses[k] = torch.cat(losses[k]).mean(dim=0)
#         return losses


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
    criterion = tc.distill_criterion_type()
    while tc.state.current_batch <= tc.last_batch:
        # save model conditionally
        if tc.accelerator.is_main_process:
            now = datetime.now()
            if (now - model_saved).total_seconds() > 60 * args.save_every:
                save_state(tc.accelerator, tc.state_path)
                model_saved = datetime.now()
        # model evaluation
        in_training_eval(args, tc)
        # batch preparation
        try:
            X, y = next(train_iter)
        except StopIteration:
            train_iter = iter(tc.train_loader)
            X, y = next(train_iter)
        if mixup_fn is not None:
            X, y = mixup_fn(X, y)
        # training step
        # (save inputs and outputs with hooks)
        set_for_distillation_iteration(tc)
        with torch.no_grad():
            tc.base_model(X)
        # iterate over each layer, calculate loss for each ACM individually
        module_losses = []
        for module_name in tc.replaced_module_names:
            module = tc.replacement_modules[module_name]
            original_input = tc.base_modules_inputs[module_name][0].detach()
            original_output = tc.base_modules_outputs[module_name].detach()
            output = module(original_input)
            assert not torch.any(
                torch.isnan(output)), f'NaN present in {module_name} output of batch {tc.state.current_batch}'
            module_loss = criterion(output, original_output)
            assert not torch.any(
                torch.isnan(module_loss)), f'NaN present in {module_name} loss for batch {tc.state.current_batch}'
            module_losses.append(module_loss)
            del original_input, original_output, output
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/Module {module_name} loss', module_loss.item(),
                                     global_step=tc.state.current_batch)
        distill_loss = torch.stack(module_losses).mean()
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Train/Rep. Distill Loss', distill_loss.item(), global_step=tc.state.current_batch)
            # get activations / pre-activations and compute sparsity loss
        sparsity_loss = 0.0
        sparse_activations_sum = 0.0
        total_activations_sum = 0.0
        # warning! inputs (e.g. pre-activations for gelu) are tuples, outputs (e.g. relu activations) are tensors
        if args.dsti_enforce_mode == 'gelu_preactivations_hoyer_clamped':
            for module_name, preacts in tc.sparsity_modules_inputs.items():
                # get first input tensor from the input tuple
                preacts = preacts[0]
                clamped_preactivations = preacts.clamp(
                    min=args.dsti_clamp_displacement) - args.dsti_clamp_displacement
                sparsity_loss += square_hoyer(clamped_preactivations, dim=-1).mean()
            sparsity_loss /= len(tc.sparsity_modules_inputs)
        elif args.dsti_enforce_mode == 'gelu_preactivations_clamped':
            for module_name, preacts in tc.sparsity_modules_inputs.items():
                # get first input tensor from the input tuple
                preacts = preacts[0]
                clamped_preactivations = preacts.clamp(
                    min=args.dsti_clamp_displacement) - args.dsti_clamp_displacement
                sparsity_loss += clamped_preactivations.sum(dim=-1).mean()
            sparsity_loss /= len(tc.sparsity_modules_inputs)
        elif args.dsti_enforce_mode == 'relu_hoyer':
            for module_name, acts in tc.sparsity_modules_outputs.items():
                sparsity_loss += square_hoyer(acts, dim=-1).mean()
                sparse_activations_sum += (acts <= 0).sum().item()
                total_activations_sum += acts.numel()
            sparsity_loss /= len(tc.sparsity_modules_inputs)
        elif args.dsti_enforce_mode == 'relu_l1':
            for module_name, acts in tc.sparsity_modules_outputs.items():
                sparsity_loss += acts.sum(dim=-1).mean()
                sparse_activations_sum += (acts <= 0).sum().item()
                total_activations_sum += acts.numel()
            sparsity_loss /= len(tc.sparsity_modules_inputs)
        if args.dsti_enforce_schedule == 'linear':
            dsti_enforce_weight = tc.state.current_batch / tc.last_batch * args.dsti_enforce_weight
            if tc.accelerator.is_main_process and tc.state.current_batch in tc.eval_batch_list:
                tc.writer.add_scalar('Train/Enforce weight', dsti_enforce_weight, global_step=tc.state.current_batch)
        elif args.dsti_enforce_schedule is None:
            dsti_enforce_weight = args.dsti_enforce_weight if args.dsti_enforce_weight is not None else 0.0
        loss = distill_loss + dsti_enforce_weight * sparsity_loss
        tc.optimizer.zero_grad(set_to_none=True)
        tc.accelerator.backward(loss)
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
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Train/Loss', loss.item(), global_step=tc.state.current_batch)
            if args.dsti_enforce_mode is not None:
                tc.writer.add_scalar('Train/Sparsity loss', sparsity_loss.item(), global_step=tc.state.current_batch)
                sparsity = sparse_activations_sum / total_activations_sum
                tc.writer.add_scalar('Train/Sparsity', sparsity, global_step=tc.state.current_batch)
                tc.writer.add_scalar('Train/Progress', tc.state.current_batch / tc.last_batch, global_step=tc.state.current_batch)
        # bookkeeping
        tc.state.current_batch += 1


def train(args):
    configure_logging()
    tc = ReplaceModulesTrainingContext()
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_data(args, tc)
    setup_model(args, tc)
    setup_for_training(args, tc)
    setup_optimization(args, tc)
    setup_state(tc)
    training_loop(args, tc)
    final_eval(args, tc)


def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()