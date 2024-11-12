import logging

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm

from common import get_default_args
from eval import benchmark_moe
from methods.dynamic_sparsification.train_routers import set_for_eval_with_topk, set_for_eval_with_dynk
from train import TrainingContext, setup_accelerator, setup_data, setup_optimization, setup_files_and_logging, \
    setup_state
from utils import load_model, save_state, remove_hooks, save_final


def setup_model(args, tc):
    assert args.model_class == 'dsti_fewshot_eval'
    base_model, base_args, _ = load_model(args, args.base_on, args.exp_id)
    tc.model = tc.accelerator.prepare(base_model)


def online_evaluate_moe(accelerator: Accelerator,
                        model: torch.nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        criterion_class: torch.nn.Module,
                        cost_without_experts,
                        token_expert_costs,
                        batches: int = 0,
                        return_counts: bool = False):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    total_average_flops = cost_without_experts
    executed_expert_tokens = {name: 0 for name in token_expert_costs.keys()}
    total_expert_tokens = {name: 0 for name in token_expert_costs.keys()}
    expert_average_costs = {}
    true_token_idx = torch.tensor(data_loader.dataset.true_token_idx, device=accelerator.device)
    false_token_idx = torch.tensor(data_loader.dataset.false_token_idx, device=accelerator.device)
    if batches is not None:
        total = batches
    else:
        total = len(data_loader)
    with torch.inference_mode():
        for batch, (X, y) in enumerate(tqdm(data_loader, total=total, desc='Running evaluation...')):
            y_pred, gating_data = model(X, return_gating_data=True)
            # each element of gating_data_list is a tuple
            # because different MoEs classes can return more than simply the gating network's final outputs
            # we select only the final routing decisions
            gating_data = {k: v[0] for k, v in gating_data.items()}
            # gating data should be a dict with tensor values of size (batch_size, sequence_length, num_experts) now
            y_pred, y, gating_data = accelerator.gather_for_metrics((y_pred, y, gating_data))
            probs_true = y_pred[:, -1, true_token_idx]
            probs_false = y_pred[:, -1, false_token_idx]
            preds_true = probs_true > probs_false
            target = y == true_token_idx
            hits = preds_true == target
            correct += hits.sum().item()
            total += y.numel()  # use numel since targets can be batches of sequences
            for moe_name in token_expert_costs.keys():
                executed_expert_tokens[moe_name] += (gating_data[moe_name] > 0.0).long().sum().item()
                total_expert_tokens[moe_name] += gating_data[moe_name].numel()
            if batches > 0 and batch == batches - 1:
                break
    for moe_name, token_expert_cost in token_expert_costs.items():
        expert_average_cost = executed_expert_tokens[moe_name] * token_expert_cost / total
        logging.info(f'Averaged FLOPs for MoE {moe_name}: {expert_average_cost}')
        expert_average_costs[moe_name] = expert_average_cost
        total_average_flops += expert_average_cost
    logging.info(f'Model accuracy: {correct / total}')
    # loss, acc
    if return_counts:
        return running_loss / total, correct / total, total_average_flops, expert_average_costs, \
            executed_expert_tokens, total_expert_tokens
    else:
        return running_loss / total, correct / total, total_average_flops, expert_average_costs


def evaluate_static_model(accelerator: Accelerator,
                          model: torch.nn.Module,
                          data_loader: torch.utils.data.DataLoader,
                          criterion_class: torch.nn.Module,
                          batches: int = 0):
    criterion = criterion_class(reduction='sum')
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    true_token_idx = torch.tensor(data_loader.dataset.true_token_idx, device=accelerator.device)
    false_token_idx = torch.tensor(data_loader.dataset.false_token_idx, device=accelerator.device)
    if batches is not None:
        total = batches
    else:
        total = len(data_loader)
    with torch.inference_mode():
        for batch, (X, y) in enumerate(tqdm(data_loader, total=total, desc='Running evaluation...')):
            y_pred = model(X)
            # each element of gating_data_list is a tuple
            # because different MoEs classes can return more than simply the gating network's final outputs
            # we select only the final routing decisions
            # gating data should be a dict with tensor values of size (batch_size, sequence_length, num_experts) now
            y_pred, y = accelerator.gather_for_metrics((y_pred, y))
            probs_true = y_pred[:, -1, true_token_idx]
            probs_false = y_pred[:, -1, false_token_idx]
            preds_true = probs_true > probs_false
            target = y == true_token_idx
            hits = preds_true == target
            correct += hits.sum().item()
            total += y.numel()  # use numel since targets can be batches of sequences
            if batches > 0 and batch == batches - 1:
                break
    logging.info(f'Model accuracy: {correct / total}')
    return running_loss / total, correct / total


def final_eval(args, tc):
    if not tc.final_path.exists():
        if tc.accelerator.is_main_process:
            save_state(tc.accelerator, tc.state_path)
        if hasattr(tc, 'hook_handles'):
            remove_hooks(tc.hook_handles)
        if hasattr(args, 'static_eval') and args.static_eval:
            loss, acc = evaluate_static_model(tc.accelerator, tc.model, tc.test_loader, tc.criterion_type)
        else:
            set_for_eval_with_topk(tc, 1)
            unwrapped_model = tc.accelerator.unwrap_model(tc.model)
            del tc.optimizer
            # Fix sequence length to match the flops of different evaluations in case we use no padding
            org_seq_length = tc.test_loader.dataset.seq_length
            tc.test_loader.dataset.seq_length = 128
            cost_without_experts, token_expert_costs, model_params = benchmark_moe(unwrapped_model, tc.test_loader)
            tc.test_loader.dataset.seq_length = org_seq_length
            final_scores = []
            final_losses = []
            final_flops = []
            final_expert_average_costs = []
            final_expert_utilization = []
            final_total_experts = []
            if args.dsti_tau_to_eval is None and args.k_to_eval is None:
                raise ValueError('Must specify either dsti_tau_to_eval or dsti_k_to_eval')
            if args.dsti_tau_to_eval is not None and args.k_to_eval is not None:
                raise ValueError('Must specify either dsti_tau_to_eval or dsti_k_to_eval, not both')
            if args.dsti_tau_to_eval is not None:
                for tau in args.dsti_tau_to_eval:
                    if tc.accelerator.is_main_process:
                        logging.info(f'Testing on testset for tau={tau}.')
                    set_for_eval_with_dynk(tc, tau)
                    test_loss, test_acc, total_average_flops, expert_average_costs, executed_expert_tokens, total_expert_tokens = online_evaluate_moe(
                        tc.accelerator,
                        tc.model,
                        tc.test_loader,
                        tc.criterion_type,
                        cost_without_experts,
                        token_expert_costs,
                        batches=args.test_batches,
                        return_counts=True)
                    if tc.accelerator.is_main_process:
                        final_losses.append(test_loss)
                        final_scores.append(test_acc)
                        # benchmark model efficiency
                        final_flops.append(total_average_flops)
                        final_expert_average_costs.append(expert_average_costs)
                        final_expert_utilization.append(executed_expert_tokens)
                        final_total_experts.append(total_expert_tokens)
                        # log
                        tc.writer.add_scalar(f'Eval with tau={tau}/Test loss', test_loss,
                                             global_step=tc.state.current_batch)
                        tc.writer.add_scalar(f'Eval with tau={tau}/Test accuracy', test_acc,
                                             global_step=tc.state.current_batch)
                        tc.writer.add_scalar(f'Eval with tau={tau}/Model FLOPs', total_average_flops,
                                             global_step=tc.state.current_batch)
            if args.k_to_eval is not None:
                for k_to_use in args.k_to_eval:
                    if tc.accelerator.is_main_process:
                        logging.info(f'Testing on testset for k={k_to_use}.')
                    set_for_eval_with_topk(tc, k_to_use)
                    test_loss, test_acc, total_average_flops, expert_average_costs, executed_expert_tokens, total_expert_tokens = online_evaluate_moe(
                        tc.accelerator,
                        tc.model,
                        tc.test_loader,
                        tc.criterion_type,
                        cost_without_experts,
                        token_expert_costs,
                        batches=args.test_batches,
                        return_counts=True)
                    if tc.accelerator.is_main_process:
                        final_losses.append(test_loss)
                        final_scores.append(test_acc)
                        # benchmark model efficiency
                        final_flops.append(total_average_flops)
                        final_expert_average_costs.append(expert_average_costs)
                        final_expert_utilization.append(executed_expert_tokens)
                        final_total_experts.append(total_expert_tokens)
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
                final_results['hyperparam_values'] = args.dsti_tau_to_eval
                final_results['final_scores'] = final_scores
                final_results['final_flops'] = final_flops
                final_results['model_params'] = dict(model_params)
                final_results['expert_average_costs'] = final_expert_average_costs
                final_results['expert_utilization'] = final_expert_utilization
                final_results['total_experts_used'] = final_total_experts
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
    set_seed(0)
    tc = TrainingContext()
    if args.dsti_quantization is not None:
        args.cpu = True
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_model(args, tc)
    setup_data(args, tc)
    setup_optimization(args, tc)
    setup_state(tc)
    final_eval(args, tc)


def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()
