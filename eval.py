import logging
from collections import defaultdict
from typing import List, Tuple, Dict

import torch
from accelerate import Accelerator
from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table
from sklearn.metrics import roc_auc_score
from torch.nn import MultiheadAttention, LayerNorm
from torchvision.models.vision_transformer import MLP

from architectures.early_exits.pbee import PBEE
from architectures.vit import MLP as CustomMLP
from utils import flop_count, get_module_by_name, remove_hooks, find_module_names, add_save_activations_hook


def test_classification(accelerator: Accelerator,
                        model: torch.nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        criterion_class: torch.nn.Module,
                        batches: int = 0,
                        tc=None) -> Tuple[float, float]:
    criterion = criterion_class(reduction='sum')
    model.eval()
    with torch.inference_mode():
        sparse_activations, total_activations = 0., 0.
        running_loss = 0.0
        correct, total = 0, 0
        for batch, (X, y) in enumerate(data_loader):
            y_pred = model(X)
            y_pred, y = accelerator.gather_for_metrics((y_pred, y))
            y_pred_max = y_pred.argmax(dim=-1)
            if len(y_pred.shape) == 3:
                # For CE loss on sequences
                y_pred = y_pred.transpose(1, 2)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            correct += (y_pred_max == y).sum().item()
            # Again account for multi-dimensional targets
            total += y.numel()
            if hasattr(tc, 'sparsity_enforcement_mode'):
                if 'relu' in tc.sparsity_enforcement_mode:
                    for module_name, acts in tc.modules_outputs.items():
                        sparse_activations += (acts <= 0).sum().item()
                        total_activations += acts.numel()
                elif 'gelu' in tc.sparsity_enforcement_mode:
                    displacement = tc.sparsity_enforcement_displacement
                    for module_name, preacts in tc.modules_inputs.items():
                        # get first input tensor from the input tuple
                        assert isinstance(preacts, tuple)
                        preacts = preacts[0]
                        clamped_preactivations = preacts.clamp(min=displacement) - displacement
                        sparse_activations += (clamped_preactivations <= 0).sum().item()
                        total_activations += preacts.numel()
                else:
                    raise NotImplementedError()

            if batches > 0 and batch == batches - 1:
                break
    
    if hasattr(tc, 'sparsity_enforcement_mode'):
        tc.sparsity = sparse_activations / total_activations

    # loss, acc
    return running_loss / total, correct / total


def get_preds(accelerator: Accelerator,
              model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              batches: int = 0):
    model.eval()
    batch_outputs = []
    batch_labels = []
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            output = model(X)
            output, y = accelerator.gather_for_metrics((output, y))
            batch_outputs.append(output.detach().cpu())
            batch_labels.append(y.detach().cpu())
            if batches > 0 and batch == batches - 1:
                break
    batch_outputs = torch.cat(batch_outputs)
    batch_labels = torch.cat(batch_labels)
    return batch_outputs, batch_labels


def get_preds_earlyexiting(accelerator: Accelerator,
                           model: torch.nn.Module,
                           data_loader: torch.utils.data.DataLoader,
                           batches: int = 0):
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    batch_outputs = []
    batch_labels = []
    with torch.inference_mode():
        unwrapped_model.all_mode()
        for batch, (X, y) in enumerate(data_loader):
            output = model(X)
            output, y = accelerator.gather_for_metrics((output, y))
            y_preds = [y_pred.detach().cpu() for y_pred in output]
            batch_outputs.append(y_preds)
            batch_labels.append(y.detach().cpu())
            if batches > 0 and batch == batches - 1:
                break
    batch_head_preds = []
    for i in range(unwrapped_model.number_of_heads):
        head_outputs = torch.cat([batch_output[i] for batch_output in batch_outputs])
        batch_head_preds.append(head_outputs)
    batch_labels = torch.cat(batch_labels)
    return batch_head_preds, batch_labels


def get_preds_moe(accelerator: Accelerator,
                  model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  batches: int = 0):
    model.eval()
    batch_outputs = []
    batch_labels = []
    batch_gating_data = defaultdict(list)
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            output, gating_data = model(X, return_gating_data=True)
            # we select only the final routing decisions
            gating_data = {k: v[0] for k, v in gating_data.items()}
            output, y, gating_data = accelerator.gather_for_metrics((output, y, gating_data))
            batch_outputs.append(output.detach().cpu())
            batch_labels.append(y.detach().cpu())
            for k, v in gating_data.items():
                batch_gating_data[k].append(v.detach().cpu())
            if batches > 0 and batch == batches - 1:
                break
    batch_outputs = torch.cat(batch_outputs)
    batch_labels = torch.cat(batch_labels)
    for k in batch_gating_data.keys():
        batch_gating_data[k] = torch.cat(batch_gating_data[k])
    return batch_outputs, batch_labels, batch_gating_data


def online_evaluate_moe(accelerator: Accelerator,
                        model: torch.nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        criterion_class: torch.nn.Module,
                        cost_without_experts,
                        token_expert_costs,
                        batches: int = 0):
    criterion = criterion_class(reduction='sum')
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    total_average_flops = cost_without_experts
    moe_executed_tokens = {name: 0 for name in token_expert_costs.keys()}
    expert_average_costs = {}
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            y_pred, gating_data = model(X, return_gating_data=True)
            # each element of gating_data_list is a tuple
            # because different MoEs classes can return more than simply the gating network's final outputs
            # we select only the final routing decisions
            gating_data = {k: v[0] for k, v in gating_data.items()}
            # gating data should be a dict with tensor values of size (batch_size, sequence_length, num_experts) now
            y_pred, y, gating_data = accelerator.gather_for_metrics((y_pred, y, gating_data))
            y_pred_max = y_pred.argmax(dim=-1)
            if len(y_pred.shape) == 3:
                # For CE loss on sequences
                y_pred = y_pred.transpose(1, 2)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            correct += (y_pred_max == y).sum().item()
            total += y.numel() # use numel since targets can be batches of sequences
            for moe_name in token_expert_costs.keys():
                moe_executed_tokens[moe_name] += (gating_data[moe_name] > 0.0).long().sum().item()
            if batches > 0 and batch == batches - 1:
                break
    for moe_name, token_expert_cost in token_expert_costs.items():
        expert_average_cost = moe_executed_tokens[moe_name] * token_expert_cost / total
        logging.info(f'Averaged FLOPs for MoE {moe_name}: {expert_average_cost}')
        expert_average_costs[moe_name] = expert_average_cost
        total_average_flops += expert_average_cost
    # loss, acc
    return running_loss / total, correct / total, total_average_flops, expert_average_costs


def test_earlyexiting_classification(accelerator: Accelerator,
                                     model: torch.nn.Module,
                                     data_loader: torch.utils.data.DataLoader,
                                     criterion_class: torch.nn.Module,
                                     batches: int = 0):
    criterion = criterion_class(reduction='mean')
    head_preds, ys = get_preds_earlyexiting(accelerator, model, data_loader, batches)
    head_losses = [criterion(preds, ys) for preds in head_preds]
    head_accuracies = [(preds.argmax(dim=1) == ys).sum().item() / ys.size(0) for preds in head_preds]
    return head_losses, head_accuracies


def average_earlyexiting_flops(head_costs: List, head_exit_counts: torch.Tensor):
    assert len(head_costs) == head_exit_counts.size(0), f'{head_costs=}\n{head_exit_counts=}'
    total_cost = 0.0
    for h_i, h_c in enumerate(head_costs):
        total_cost += h_c * head_exit_counts[h_i].item()
    average_cost = total_cost / head_exit_counts.sum().item()
    return average_cost


def average_moe_flops(cost_without_experts, token_expert_costs, gating_data):
    assert len(token_expert_costs) == gating_data.size(1), f'{len(token_expert_costs)=}\n{len(gating_data)=}'
    total_average_flops = cost_without_experts
    expert_average_costs = {}
    for moe_name, token_expert_cost in token_expert_costs.items():
        g_x = gating_data[moe_name]
        # g_x should have a (batch_size, sequence_length, expert_num)
        assert g_x.dim() == 3
        expert_average_cost = (g_x > 0.0).long().sum().item() * token_expert_cost / g_x.size(0)
        expert_average_costs.append(expert_average_cost)
        logging.info(f'Averaged FLOPs for MoE {moe_name}: {expert_average_cost}')
        total_average_flops += expert_average_cost
    return total_average_flops, expert_average_costs


def evaluate_earlyexiting_classification(model: torch.nn.Module,
                                         head_preds: List[torch.Tensor],
                                         labels: torch.Tensor,
                                         head_costs: List[FlopCountAnalysis],
                                         eval_thresholds: int) -> Dict:
    head_accuracies = []
    for i, head_pred in enumerate(head_preds):
        head_accuracy = (head_pred.argmax(dim=1) == labels).sum().item() / labels.size(0)
        head_accuracies.append(head_accuracy)
    head_flops = [head_cost.total() for head_cost in head_costs]
    thresholds = torch.linspace(0.0, 1.0, steps=eval_thresholds, device=labels.device)
    threshold_accuracies = []
    threshold_flops = []
    # separate path for evaluating PBEE
    if isinstance(model, PBEE):
        patience_thresholds = torch.arange(0, len(head_preds), device=labels.device)
        for patience_threshold in patience_thresholds:
            exit_at = torch.zeros_like(labels) - 1
            outputs = torch.zeros_like(head_preds[0])
            # start from second head, set patience to one after first head
            prev_answers = torch.zeros_like(labels) - 1
            patience = torch.zeros_like(prev_answers)
            for i, head_pred in enumerate(head_preds):
                patience = torch.where(head_pred.argmax(-1) == prev_answers, patience + 1, 1)
                unresolved_mask = exit_at == -1
                exit_mask = (patience > patience_threshold) & unresolved_mask
                exit_at[exit_mask] = i
                outputs[exit_mask] = head_pred[exit_mask]
                prev_answers = head_pred.argmax(-1)
            unresolved_mask = exit_at == -1
            outputs[unresolved_mask] = head_preds[-1][unresolved_mask]
            exit_at[unresolved_mask] = len(head_preds) - 1
            threshold_accuracy = ((outputs.argmax(dim=-1) == labels).sum() / labels.size(0)).item()
            exits_bincounted = exit_at.bincount(minlength=len(head_preds))
            threshold_cost = average_earlyexiting_flops(head_flops, exits_bincounted)
            threshold_accuracies.append(threshold_accuracy)
            threshold_flops.append(threshold_cost)
    else:
        head_probs = [preds.softmax(dim=-1) for preds in head_preds]
        thresholds = torch.linspace(0.0, 1.0, steps=eval_thresholds, device=labels.device)
        for threshold in thresholds:
            exit_at = torch.zeros_like(labels) - 1
            outputs = torch.zeros_like(head_probs[0])
            for i, head_prob in enumerate(head_probs):
                head_confidences, _ = head_prob.max(dim=-1)
                unresolved_mask = exit_at == -1
                exit_mask = (head_confidences > threshold) & unresolved_mask
                exit_at[exit_mask] = i
                outputs[exit_mask] = head_prob[exit_mask]
            unresolved_mask = exit_at == -1
            outputs[unresolved_mask] = head_probs[-1][unresolved_mask]
            exit_at[unresolved_mask] = len(head_probs) - 1
            threshold_accuracy = ((outputs.argmax(dim=-1) == labels).sum() / labels.size(0)).item()
            exits_bincounted = exit_at.bincount(minlength=len(head_probs))
            threshold_cost = average_earlyexiting_flops(head_flops, exits_bincounted)
            threshold_accuracies.append(threshold_accuracy)
            threshold_flops.append(threshold_cost)
    results = {'head_scores': head_accuracies, 'head_flops': head_flops, 'thresholds': thresholds,
               'threshold_scores': threshold_accuracies, 'threshold_flops': threshold_flops}
    return results


def evaluate_classification(preds: torch.Tensor, labels: torch.Tensor, criterion_class: torch.nn.Module):
    criterion = criterion_class(reduction='mean')
    preds_max = preds.argmax(dim=1)
    loss = criterion(preds, labels).item()
    accuracy = (preds_max == labels).double().mean().item()
    return loss, accuracy


def ks_calibration_error(probs, labels):
    '''https://arxiv.org/abs/2006.12800'''
    assert probs.dim() == 2, f'{probs.size()=}'
    num_classes = probs.size(-1)
    labels_oh = torch.nn.functional.one_hot(labels, num_classes)
    num_samples = probs.size(0)
    ks_errors = [0.0] * num_classes
    for k in range(num_classes):
        class_probs = probs[..., k]
        class_labels = labels_oh[..., k]
        sorted_probs, indices = class_probs.sort()
        h_tilde = torch.cumsum(class_labels[indices] / num_samples, dim=0)
        h = torch.cumsum(sorted_probs / num_samples, dim=0)
        ks_errors[k] += (h - h_tilde).abs().max().item()
    # TODO is averaging appropriate?
    ks_error = sum(ks_errors) / num_classes
    return ks_error, ks_errors


def evaluate_calibration(preds: torch.Tensor,
                         labels: torch.Tensor) -> Dict:
    probs = preds.softmax(dim=-1)
    # ignores per-class calibration scores, takes the average
    calibration_score, _ = ks_calibration_error(probs, labels)
    results = {'final_score': calibration_score}
    return results


# TODO possibly generalize this code and merge it with accuracy and ood
def evaluate_earlyexiting_calibration(head_preds: List[torch.Tensor],
                                      labels: torch.Tensor,
                                      head_costs: List[int],
                                      thresholds: torch.Tensor) -> Dict:
    head_probs = [preds.softmax(dim=-1) for preds in head_preds]
    head_calibration_scores = []
    for i, head_prob in enumerate(head_probs):
        head_calibration_score, _ = ks_calibration_error(head_prob, labels)
        head_calibration_scores.append(head_calibration_score)
    threshold_calibration_scores = []
    threshold_flops = []
    for threshold in thresholds:
        exit_at = torch.zeros_like(labels) - 1
        outputs = torch.zeros_like(head_probs[0])
        for i, head_prob in enumerate(head_probs):
            head_confidences, _ = head_prob.max(dim=-1)
            unresolved_mask = exit_at == -1
            exit_mask = (head_confidences > threshold) & unresolved_mask
            exit_at[exit_mask] = i
            outputs[exit_mask] = head_prob[exit_mask]
        unresolved_mask = exit_at == -1
        outputs[unresolved_mask] = head_probs[-1][unresolved_mask]
        exit_at[unresolved_mask] = len(head_probs) - 1
        threshold_calibration_score, _ = ks_calibration_error(outputs, labels)
        exits_bincounted = exit_at.bincount(minlength=len(head_probs))
        threshold_cost = average_earlyexiting_flops(head_costs, exits_bincounted)
        threshold_calibration_scores.append(threshold_calibration_score)
        threshold_flops.append(threshold_cost)
    results = {'head_scores': head_calibration_scores, 'head_flops': head_costs, 'thresholds': thresholds,
               'threshold_scores': threshold_calibration_scores, 'threshold_flops': threshold_flops}
    return results


def evaluate_ood_detection(id_preds: List[torch.Tensor],
                           ood_preds: torch.Tensor) -> Dict:
    id_confidences = id_preds.softmax(dim=-1).max(dim=-1)[0]
    ood_confidences = ood_preds.softmax(dim=-1).max(dim=-1)[0]
    confidences = torch.cat([id_confidences, ood_confidences])
    ood_labels = torch.cat([torch.ones_like(id_confidences), torch.zeros_like(ood_confidences)])
    ood_score = roc_auc_score(ood_labels.cpu().numpy(), confidences.cpu().numpy())
    assert 0.0 <= ood_score <= 1.0, f'AUROC: {ood_score}'
    results = {'final_score': ood_score}
    return results


def evaluate_earlyexiting_ood_detection(head_id_preds: List[torch.Tensor],
                                        head_ood_preds: List[torch.Tensor],
                                        head_costs: List[int],
                                        thresholds: torch.Tensor) -> Dict:
    # TODO this assumes the head costs are the same for the OOD dataset - add support for different costs
    head_id_confidences = [preds.softmax(dim=-1).max(dim=-1)[0] for preds in head_id_preds]
    head_ood_confidences = [preds.softmax(dim=-1).max(dim=-1)[0] for preds in head_ood_preds]
    head_confidences = [torch.cat([id_confidences, ood_confidences]) for id_confidences, ood_confidences in
                        zip(head_id_confidences, head_ood_confidences)]
    ood_labels = torch.cat([torch.ones_like(head_id_confidences[0], dtype=torch.int),
                            torch.zeros_like(head_ood_confidences[0], dtype=torch.int)])
    head_ood_scores = []
    for i, head_confs in enumerate(head_confidences):
        head_ood_score = roc_auc_score(ood_labels.cpu().numpy(), head_confs.cpu().numpy())
        head_ood_scores.append(head_ood_score)
    threshold_ood_scores = []
    threshold_flops = []
    for threshold in thresholds:
        exit_at = torch.zeros_like(ood_labels) - 1
        outputs = torch.zeros_like(head_confidences[0])
        for i, head_confs in enumerate(head_confidences):
            unresolved_mask = exit_at == -1
            exit_mask = (head_confs > threshold) & unresolved_mask
            exit_at[exit_mask] = i
            outputs[exit_mask] = head_confs[exit_mask]
        unresolved_mask = exit_at == -1
        outputs[unresolved_mask] = head_confidences[-1][unresolved_mask]
        exit_at[unresolved_mask] = len(head_confidences) - 1
        threshold_ood_detection_score = roc_auc_score(ood_labels.cpu().numpy(), outputs.cpu().numpy())
        exits_bincounted = exit_at.bincount(minlength=len(head_confidences))
        threshold_cost = average_earlyexiting_flops(head_costs, exits_bincounted)
        threshold_ood_scores.append(threshold_ood_detection_score)
        threshold_flops.append(threshold_cost)
    results = {'head_scores': head_ood_scores, 'head_flops': head_costs, 'thresholds': thresholds,
               'threshold_scores': threshold_ood_scores, 'threshold_flops': threshold_flops}
    return results


def benchmark(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader) -> Tuple[FlopCountAnalysis, Dict]:
    model.eval()
    # workaround for the missing implementation of 'aten::_native_multi_head_attention' flop counter
    for m in model.modules():
        if isinstance(m, MultiheadAttention):
            m.train()
    #
    X, _ = next(iter(data_loader))
    if isinstance(X, dict):
        sample = {k: v[:1] for k, v in X.items()}
    else:
        sample = X[:1]
    with torch.inference_mode():
        model_costs = flop_count(model, (sample,))
        param_count = parameter_count(model)
    logging.info(f'Ops by operator:\n{model_costs.by_operator()}')
    logging.info(f'Ops by module:\n{flop_count_table(model_costs, max_depth=7)}')
    logging.info(f'Total ops: {model_costs.total()}')
    unsupported = model_costs.unsupported_ops()
    if len(unsupported) > 0:
        for k, v in unsupported.items():
            logging.warning(f'Unsupported op: {k} (occurrences: {v})')
    uncalled = model_costs.uncalled_modules()
    if len(uncalled) > 0:
        for m in uncalled:
            logging.warning(f'Uncalled module: {m}')
    return model_costs, param_count


def benchmark_earlyexiting(model: torch.nn.Module,
                           data_loader: torch.utils.data.DataLoader) \
        -> Tuple[List[FlopCountAnalysis], Dict]:
    model.eval()
    # workaround for the missing implementation of 'aten::_native_multi_head_attention' flop counter
    for m in model.modules():
        if isinstance(m, MultiheadAttention):
            m.train()
    #
    X, _ = next(iter(data_loader))
    if isinstance(X, dict):
        sample = {k: v[0].unsqueeze(0) for k, v in X.items()}
    else:
        sample = X[0].unsqueeze(0)
    with torch.inference_mode():
        param_count = parameter_count(model)
        head_costs = []
        for head_i in range(model.number_of_heads):
            model.select_head(head_i)
            head_costs.append(flop_count(model, (sample,)))
            logging.info(f'Ops for head {head_i}: {head_costs[head_i].total()}')
    unsupported = head_costs[-1].unsupported_ops()
    if len(unsupported) > 0:
        for k, v in unsupported.items():
            logging.warning(f'Unsupported op: {k} (occurrences: {v})')
    uncalled = head_costs[-1].uncalled_modules()
    if len(uncalled) > 0:
        for m in uncalled:
            logging.warning(f'Uncalled module: {m}')
    return head_costs, param_count


def benchmark_moe(model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader):
    from architectures.moe.moe_layers import MoELayer, ExecuteAllExperts, ModuleBatchedExperts
    model_costs, model_params = benchmark(model, data_loader)
    # find MoE modules and order them
    moe_module_names = find_module_names(model, lambda _, m: isinstance(m, MoELayer))
    # add hooks on gating networks and expert modules
    experts_module_names = {}
    for moe_module_name in moe_module_names:
        moe_module = get_module_by_name(model, moe_module_name)
        # find the experts module
        experts_names = find_module_names(moe_module,
                                          lambda _, m: isinstance(m, (ExecuteAllExperts, ModuleBatchedExperts)))
        assert len(experts_names) == 1, f'{len(experts_names)=}'
        experts_module_names[moe_module_name] = f'{moe_module_name}.{experts_names[0]}'
    # add hooks
    expert_module_name_list = [v for v in experts_module_names.values()]
    # expert_to_module_mapping = {v: k for k, v in experts_module_names.items()}
    experts_inputs, _, experts_handles = add_save_activations_hook(model, expert_module_name_list)
    # push an example though forward
    X, _ = next(iter(data_loader))
    if isinstance(X, dict):
        sample = {k: v[:1] for k, v in X.items()}
    else:
        sample = X[:1]
    _ = model(sample).detach()
    # push a single sample though each of the modules and calculate its costs
    cost_without_experts = model_costs.total()
    expert_costs = {}
    for moe_name in moe_module_names:
        experts_name = experts_module_names[moe_name]
        experts_module = get_module_by_name(model, experts_name)
        # calculate cost of the gating network
        gating_cost = model_costs.by_module()[moe_name] - model_costs.by_module()[experts_name]
        # calculate the cost of a single expert for a single sample
        experts_input = experts_inputs[experts_name]
        assert experts_input[0].dim() == 2
        assert experts_input[1].dim() == 2
        dummy_routing_tensor = torch.zeros_like(experts_input[1])
        dummy_routing_tensor[0, 0] = 1.0
        experts_input = (experts_input[0], dummy_routing_tensor)
        with torch.cuda.amp.autocast(dtype=experts_input[1].dtype):
            token_expert_cost = flop_count(experts_module, experts_input).total()
        if isinstance(experts_module, ExecuteAllExperts):
            # in this case all experts for each token in the sequence are executed
            # so we divide by these two values
            token_expert_cost /= experts_module.num_experts * dummy_routing_tensor.size(0)
        logging.info(
            f'MoE {moe_name} single token-expert cost: {token_expert_cost}; sequence length: {experts_input[0].size(0)}')
        cost_without_experts -= model_costs.by_module()[moe_name] - gating_cost
        expert_costs[moe_name] = token_expert_cost
    remove_hooks(experts_handles)
    logging.info(f'Model cost without experts: {cost_without_experts}')
    return cost_without_experts, expert_costs, model_params
