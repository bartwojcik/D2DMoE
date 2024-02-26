import logging
from datetime import datetime
from typing import Callable, Dict, List, Type

import torch
from omegaconf import OmegaConf
from torch import nn

from architectures.moe.moefication import MoeficationMoE, add_routers
from common import INIT_NAME_MAP, LOSS_NAME_MAP, get_default_args
from eval import benchmark_moe, online_evaluate_moe
from train import (
    TrainingContext,
    setup_accelerator,
    setup_data,
    setup_files_and_logging,
    setup_optimization,
    setup_state,
)
from utils import (
    Mixup,
    add_save_inputs_hook,
    add_save_outputs_hook,
    get_lrs,
    get_module_name,
    load_model,
    remove_hooks,
    save_final,
    save_state,
)


class RouterTrainingContext(TrainingContext):
    moe_modules: Dict[str, nn.Module] = None
    captured_layer_name_map: Dict[str, str] = None
    saved_inputs: Dict = None
    saved_outputs: Dict = None
    hook_handles: List = None
    router_criterion_type: Type = None
    router_criterion: Callable = None


def setup_model(args, tc):
    model, base_args, _ = load_model(args, args.base_on, args.exp_id)
    tc.moe_modules = add_routers(model, args.model_args)
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        init_fun(tc.model)
    tc.model = tc.accelerator.prepare(model)


def set_for_train_iteration(tc):
    tc.model.eval()
    tc.model.requires_grad_(False)
    for moe_name, moe_module in tc.moe_modules.items():
        assert moe_module.router is not None
        moe_module.router.train()
        moe_module.router.requires_grad_(True)
        moe_module.forward_mode = 'all'


def get_captured_layer_name_map(model, moe_modules: Dict):
    module_names_map = {}
    for moe_name, moe_module in moe_modules.items():
        # TODO support other experts implementations than ExecuteAllExperts
        module_names_map[moe_name] = get_module_name(model, moe_module.experts.layers[0])
    return module_names_map


def setup_for_training(args, tc):
    set_for_train_iteration(tc)
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    tc.captured_layer_name_map = get_captured_layer_name_map(unwrapped_model, tc.moe_modules)
    tc.saved_inputs, input_handles = add_save_inputs_hook(unwrapped_model, tc.moe_modules.keys())
    tc.saved_outputs, output_handles = add_save_outputs_hook(unwrapped_model, tc.captured_layer_name_map.values())
    tc.hook_handles = input_handles + output_handles
    tc.router_criterion_type = LOSS_NAME_MAP[args.router_loss_type]
    criterion_args = args.router_loss_args if args.router_loss_args is not None else {}
    tc.router_criterion = tc.router_criterion_type(reduction='mean', **criterion_args)


def set_for_eval_with_gating(tc, k):
    tc.model.eval()
    for m in tc.model.modules():
        if isinstance(m, MoeficationMoE):
            m.forward_mode = 'topk'
            m.k = k


def in_training_eval(args, tc):
    if tc.state.current_batch in tc.eval_batch_list:
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Train/Progress',
                                 tc.state.current_batch / tc.last_batch,
                                 global_step=tc.state.current_batch)
        unwrapped_model = tc.accelerator.unwrap_model(tc.model)
        set_for_eval_with_gating(tc, 1)
        cost_without_experts, token_expert_costs, model_params = benchmark_moe(unwrapped_model, tc.test_loader)
        for k_to_use in args.k_to_eval:
            if tc.accelerator.is_main_process:
                logging.info(f'Testing on testset for k={k_to_use} on {args.eval_batches} batches.')
            set_for_eval_with_gating(tc, k_to_use)
            test_loss, test_acc, _, _ = online_evaluate_moe(tc.accelerator,
                                                            tc.model,
                                                            tc.test_loader,
                                                            tc.criterion_type,
                                                            cost_without_experts,
                                                            token_expert_costs,
                                                            batches=args.eval_batches)
            if tc.accelerator.is_main_process:
                logging.info(f'Testing on trainset for k={k_to_use} on {args.eval_batches} batches.')
            train_loss, train_acc, _, _ = online_evaluate_moe(tc.accelerator,
                                                              tc.model,
                                                              tc.train_loader,
                                                              tc.criterion_type,
                                                              cost_without_experts,
                                                              token_expert_costs,
                                                              batches=args.eval_batches)
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Eval with k={k_to_use}/Test loss', test_loss,
                                     global_step=tc.state.current_batch)
                tc.writer.add_scalar(f'Eval with k={k_to_use}/Test accuracy', test_acc,
                                     global_step=tc.state.current_batch)
                tc.writer.add_scalar(f'Eval with k={k_to_use}/Train loss', train_loss,
                                     global_step=tc.state.current_batch)
                tc.writer.add_scalar(f'Eval with k={k_to_use}/Train accuracy', train_acc,
                                     global_step=tc.state.current_batch)


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
        # batch preparation
        try:
            X, y = next(train_iter)
        except StopIteration:
            train_iter = iter(tc.train_loader)
            X, y = next(train_iter)
        if mixup_fn is not None:
            X, y = mixup_fn(X, y)
        # forward
        set_for_train_iteration(tc)
        with torch.no_grad():
            _ = tc.model(X)
        router_losses = []
        for moe_name, moe in tc.moe_modules.items():
            router = moe.router
            input = tc.saved_inputs[moe_name][0]
            intermediate_output = tc.saved_outputs[tc.captured_layer_name_map[moe_name]]
            with torch.no_grad():
                # assumes ReLU activation
                # intermediate_output size is (num_experts, batch_size * seq_len, expert_dim)
                assert torch.all(intermediate_output >= 0.0), f'{intermediate_output=}'
                router_label = intermediate_output.sum(dim=-1)
                router_label = router_label / router_label.max()
                router_label = router_label.view(router_label.size(0), input.size(0), input.size(1))
                router_label = router_label.permute(1, 2, 0).detach()
                assert torch.all((router_label >= 0.0) & (router_label <= 1.0)), f'{router_label}'
            router_output = router(input)
            router_loss = tc.router_criterion(router_output, router_label)
            router_losses.append(router_loss)
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/Router {moe_name} loss', router_loss.item(),
                                     global_step=tc.state.current_batch)
        loss = torch.stack(router_losses).mean()
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Train/Average loss', loss.item(), global_step=tc.state.current_batch)
        # gradient computation
        tc.optimizer.zero_grad(set_to_none=True)
        tc.accelerator.backward(loss)
        if args.clip_grad_norm is not None:
            total_norm = tc.accelerator.clip_grad_norm_(tc.model.parameters(), args.clip_grad_norm)
            if tc.accelerator.is_main_process and total_norm is not None: # TODO check if total_norm=None indicates a bug??:
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
        tc.state.current_batch += 1


def final_eval(args, tc):
    if not tc.final_path.exists():
        if tc.accelerator.is_main_process:
            save_state(tc.accelerator, tc.state_path)
        remove_hooks(tc.hook_handles)
        set_for_eval_with_gating(tc, 1)
        unwrapped_model = tc.accelerator.unwrap_model(tc.model)
        cost_without_experts, token_expert_costs, model_params = benchmark_moe(unwrapped_model, tc.test_loader)
        final_scores = []
        final_losses = []
        final_flops = []
        for k_to_use in args.k_to_test:
            if tc.accelerator.is_main_process:
                logging.info(f'Testing on testset for k={k_to_use}.')
            set_for_eval_with_gating(tc, k_to_use)
            test_loss, test_acc, total_average_flops, expert_average_costs = online_evaluate_moe(tc.accelerator,
                                                                                                 tc.model,
                                                                                                 tc.test_loader,
                                                                                                 tc.criterion_type,
                                                                                                 cost_without_experts,
                                                                                                 token_expert_costs)
            if tc.accelerator.is_main_process:
                final_losses.append(test_loss)
                final_scores.append(test_acc)
                # benchmark model efficiency
                final_flops.append(total_average_flops)
                # log
                tc.writer.add_scalar(f'Eval with k={k_to_use}/Test loss', test_loss,
                                     global_step=tc.state.current_batch)
                tc.writer.add_scalar(f'Eval with k={k_to_use}/Test accuracy', test_acc,
                                     global_step=tc.state.current_batch)
                tc.writer.add_scalar(f'Eval with k={k_to_use}/Model FLOPs', total_average_flops,
                                     global_step=tc.state.current_batch)

        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Eval/Model Params', model_params[''], global_step=tc.state.current_batch)

            final_results = {}
            final_results['args'] = args
            final_results['model_state'] = unwrapped_model.state_dict()
            final_results['test_losses'] = final_losses
            final_results['hyperparam_values'] = args.k_to_test
            final_results['final_scores'] = final_scores
            final_results['final_flops'] = final_flops
            final_results['model_params'] = dict(model_params)
            save_final(args, tc.final_path, final_results)


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
    setup_for_training(args, tc)
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
