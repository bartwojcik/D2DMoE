import logging
from datetime import datetime

import torch
from omegaconf import OmegaConf

from common import get_default_args, INIT_NAME_MAP
from eval import test_classification, benchmark_avit, get_preds_avit, evaluate_classification, average_avit_flops
from train import TrainingContext, setup_accelerator, setup_files_and_logging, setup_data, setup_optimization, \
    setup_state
from utils import save_state, get_lrs, save_final, Mixup, create_model, load_model


def get_distribution_target(length, target_depth):
    dist = torch.distributions.Normal(loc=target_depth, scale=1.0)
    return dist.log_prob(torch.arange(length) + 1)


def setup_model(args, tc):
    base_model, base_args, _ = load_model(args, args.base_on, args.exp_id)
    model_args = {'base_model': base_model, **args.model_args}
    model = create_model(args.model_class, model_args)
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        model.apply(init_fun)
    tc.model = tc.accelerator.prepare(model)


def in_training_eval(args, tc):
    if tc.state.current_batch in tc.eval_batch_list:
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Train/Progress',
                                 tc.state.current_batch / tc.last_batch,
                                 global_step=tc.state.current_batch)
        test_loss, test_acc = test_classification(tc.accelerator,
                                                  tc.model,
                                                  tc.test_loader,
                                                  tc.criterion_type,
                                                  batches=args.eval_batches)
        train_loss, train_acc = test_classification(tc.accelerator,
                                                    tc.model,
                                                    tc.train_eval_loader,
                                                    tc.criterion_type,
                                                    batches=args.eval_batches)
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Eval/Test loss', test_loss, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Eval/Test accuracy', test_acc, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Eval/Train loss', train_loss, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Eval/Train accuracy', train_acc, global_step=tc.state.current_batch)


def training_loop(args, tc):
    if tc.accelerator.is_main_process:
        model_saved = datetime.now()
    train_iter = iter(tc.train_loader)
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    # skip the last halting score for distr_prior_loss
    log_distr_target = get_distribution_target(length=unwrapped_model.depth,
                                               target_depth=args.distr_target_depth).to(device=tc.accelerator.device)
    if tc.accelerator.is_main_process:
        logging.info(f'A-ViT {log_distr_target=}')
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
        tc.model.train()
        y_pred, rho_token, token_counter, halting_scores = tc.model(X)
        assert not y_pred.isnan().any(), f'{tc.state.current_batch=} {y_pred=}'
        assert not rho_token.isnan().any(), f'{tc.state.current_batch=} {rho_token=}'
        assert not token_counter.isnan().any(), f'{tc.state.current_batch=} {token_counter=}'
        # loss computation
        task_loss = tc.criterion(y_pred, y)
        # Ponder loss
        ponder_loss = torch.mean(rho_token)
        # Distributional prior
        halting_score_distr = torch.stack(halting_scores)
        assert not halting_score_distr.isnan().any(), f'{tc.state.current_batch=} {halting_score_distr=}'
        halting_score_distr = halting_score_distr / torch.sum(halting_score_distr)
        halting_score_distr = torch.clamp(halting_score_distr, 0.001, 0.999)
        distr_prior_loss = torch.nn.functional.kl_div(halting_score_distr.log(),
                                                      log_distr_target.detach(),
                                                      reduction='batchmean',
                                                      log_target=True)
        loss = task_loss + args.ponder_loss_weight * ponder_loss + args.distr_prior_loss_weight * distr_prior_loss
        assert not loss.isnan().any(), f'{tc.state.current_batch=} {loss=}'
        # gradient computation and training step
        tc.optimizer.zero_grad(set_to_none=True)
        tc.accelerator.backward(loss)
        if args.clip_grad_norm is not None:
            total_norm = tc.accelerator.clip_grad_norm_(tc.model.parameters(), args.clip_grad_norm)
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/Gradient norm', total_norm.item(), global_step=tc.state.current_batch)
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
            tc.writer.add_scalar(f'Train/Loss', loss.item(), global_step=tc.state.current_batch)
            tc.writer.add_scalar(f'Train/Task loss', task_loss.item(), global_step=tc.state.current_batch)
            tc.writer.add_scalar(f'Train/Ponder loss', ponder_loss.item(), global_step=tc.state.current_batch)
            tc.writer.add_scalar(f'Train/Distributional prior loss', distr_prior_loss.item(),
                                 global_step=tc.state.current_batch)
            tc.writer.add_scalar('Train/Token depth mean', token_counter.float().mean().item(), tc.state.current_batch)
            tc.writer.add_scalar('Train/Token depth max', token_counter.max().item(), tc.state.current_batch)
            tc.writer.add_scalar('Train/Token depth min', token_counter.min().item(), tc.state.current_batch)
        # bookkeeping
        tc.state.current_batch += 1


def final_eval(args, tc):
    if not tc.final_path.exists():
        if tc.accelerator.is_main_process:
            save_state(tc.accelerator, tc.state_path)
        # test on testset
        preds, labels, token_counts = get_preds_avit(tc.accelerator,
                                                     tc.model,
                                                     tc.test_loader)
        if tc.accelerator.is_main_process:
            test_loss, test_acc = evaluate_classification(preds, labels, tc.criterion_type)
            final_results = {}
            final_results['args'] = args
            unwrapped_model = tc.accelerator.unwrap_model(tc.model)
            final_results['model_state'] = unwrapped_model.state_dict()
            tc.writer.add_scalar('Eval/Test loss', test_loss, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Eval/Test accuracy', test_acc, global_step=tc.state.current_batch)
            final_results['final_score'] = test_acc
            final_results['final_loss'] = test_loss
            # benchmark model efficiency
            constant_cost, mha_sequence_costs, mlp_token_cost, model_params = benchmark_avit(unwrapped_model,
                                                                                             tc.test_loader)
            total_average_flops = average_avit_flops(constant_cost, mha_sequence_costs, mlp_token_cost, token_counts)
            final_results['model_flops'] = total_average_flops
            final_results['model_params'] = dict(model_params)
            tc.writer.add_scalar('Eval/Model FLOPs', total_average_flops, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Eval/Model Params', model_params[''], global_step=tc.state.current_batch)
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
