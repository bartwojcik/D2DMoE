import logging
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePath
from typing import Any, Callable, List, Type

import torch
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from common import INIT_NAME_MAP, LOSS_NAME_MAP, OPTIMIZER_NAME_MAP, SCHEDULER_NAME_MAP, get_default_args
from data_utils.data import DATASETS_NAME_MAP
from eval import benchmark, test_classification
from utils import (
    Mixup,
    create_model,
    generate_run_name,
    get_loader,
    get_lrs,
    get_run_id,
    load_state,
    save_final,
    save_state,
    unfrozen_parameters, load_model,
)


@dataclass
class TrainingState:
    current_batch: int = 0

    def state_dict(self):
        return {'current_batch': self.current_batch}

    def load_state_dict(self, state_dict):
        self.current_batch = state_dict['current_batch']


@dataclass
class TrainingContext:
    accelerator: Accelerator = None
    model: torch.nn.Module = None
    # savable state
    state: TrainingState = None
    # data
    train_loader: torch.utils.data.DataLoader = None
    train_eval_loader: torch.utils.data.DataLoader = None
    test_loader: torch.utils.data.DataLoader = None
    last_batch: int = None
    eval_batch_list: List[int] = None
    # optimization
    criterion_type: Type = None
    criterion: Callable = None
    optimizer: torch.optim.Optimizer = None
    scheduler: Any = None
    teacher_model: torch.nn.Module = None
    distill_weight: float = None
    # files and logging
    state_path: PurePath = None
    final_path: PurePath = None
    writer: SummaryWriter = None


def setup_accelerator(args, tc):
    # do not change the split_batches argument
    # resuming a run with different resources and split_batches=False would cause the batches_per_epoch to be different
    tc.accelerator = Accelerator(split_batches=True,
                                 # overwrite the unintuitive behaviour of accelerate, see:
                                 # https://discuss.huggingface.co/t/learning-rate-scheduler-distributed-training/30453
                                 step_scheduler_with_optimizer=False,
                                 mixed_precision=args.mixed_precision, cpu=args.cpu,
                                 # find_unused_parameters finds unused parameters when using DDP
                                 # but may affect performance negatively
                                 # sometimes it happens that somehow it works with this argument, but fails without it
                                 # so - do not remove
                                 # see: https://github.com/pytorch/pytorch/issues/43259
                                 kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
                                 )


def setup_model(args, tc):
    model = create_model(args.model_class, args.model_args)
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        init_fun(model)
    tc.model = tc.accelerator.prepare(model)
    tc.model.train()


def setup_data(args, tc):
    dataset_args = {} if args.dataset_args is None else args.dataset_args
    train_data, train_eval_data, test_data = DATASETS_NAME_MAP[args.dataset](**dataset_args)
    batch_size = args.batch_size
    if args.test_batch_size is None:
        test_batch_size = batch_size
    else:
        test_batch_size = args.test_batch_size
    tc.train_loader = get_loader(train_data, batch_size, tc.accelerator, num_workers=args.num_workers)
    tc.train_eval_loader = get_loader(train_eval_data, test_batch_size, tc.accelerator, num_workers=args.num_workers)
    tc.test_loader = get_loader(test_data, test_batch_size, tc.accelerator, num_workers=args.num_workers)
    if args.last_batch is None:
        batches_per_epoch = len(tc.train_loader)
        tc.last_batch = math.ceil(args.epochs * batches_per_epoch - 1)
        if tc.accelerator.is_main_process:
            logging.info(f'{args.epochs=} {batches_per_epoch=} {tc.last_batch=}')
    else:
        tc.last_batch = args.last_batch
        if tc.accelerator.is_main_process:
            logging.info(f'{tc.last_batch=}')
    tc.eval_batch_list = [
        round(x) for x in torch.linspace(0, tc.last_batch, steps=args.eval_points, device='cpu').tolist()
    ]


def setup_optimization(args, tc):
    criterion_args = args.loss_args
    tc.criterion_type = LOSS_NAME_MAP[args.loss_type]
    tc.criterion = tc.criterion_type(reduction='mean', **criterion_args)
    optimizer_args = args.optimizer_args
    if 'selective_weight_decay' in optimizer_args and optimizer_args['selective_weight_decay']:
        param_dict = {pn: p for pn, p in tc.model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        params = [
            {'params': decay_params, 'weight_decay': optimizer_args.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        del optimizer_args['weight_decay']
        del optimizer_args['selective_weight_decay']
    else:
        params = unfrozen_parameters(tc.model)
    tc.optimizer = tc.accelerator.prepare(
        OPTIMIZER_NAME_MAP[args.optimizer_class](params, **optimizer_args))
    if args.scheduler_class is not None:
        scheduler_args = deepcopy(args.scheduler_args)
        if 'patience' in scheduler_args:
            scheduler_args['patience'] = int(scheduler_args['patience'] * tc.last_batch)
        if args.scheduler_class == 'cosine':
            scheduler_args['T_max'] = tc.last_batch
        if args.scheduler_class in ['cosine_with_warmup', 'linear', 'inverse_sqrt']:
            scheduler_args['num_training_steps'] = tc.last_batch
        tc.scheduler = tc.accelerator.prepare(SCHEDULER_NAME_MAP[args.scheduler_class](tc.optimizer, **scheduler_args))
    if args.distill_from is not None:
        teacher_model, _, _ = load_model(args, args.distill_from, args.exp_id)
        tc.teacher_model = tc.accelerator.prepare(teacher_model)
        tc.teacher_model.eval()
        tc.distill_weight = 1.0 if args.distill_weight is None else args.distill_weight
        # TODO optionally parameterize the type of loss
        tc.distill_criterion = torch.nn.KLDivLoss(log_target=True, reduction='batchmean')


def setup_files_and_logging(args, tc):
    # files setup
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    _, run_name = generate_run_name(args)
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tc.state_path = run_dir / 'state'
    tc.final_path = run_dir / 'final.pth'
    # logging setup
    if tc.accelerator.is_main_process:
        # log config
        logging.info(f'{run_name} args:\n{args}')
        if args.use_wandb:
            entity = os.environ['WANDB_ENTITY']
            project = os.environ['WANDB_PROJECT']
            run_id = get_run_id(run_name)
            wandb.tensorboard.patch(root_logdir=str(run_dir.resolve()), pytorch=True, save=False)
            if run_id is not None:
                wandb.init(entity=entity, project=project, id=run_id, resume='must', dir=str(run_dir.resolve()))
            else:
                wandb.init(entity=entity, project=project, config=dict(args), name=run_name,
                           dir=str(run_dir.resolve()))
            wandb.run.log_code('.', include_fn=lambda path: path.endswith('.py'))
        tc.writer = SummaryWriter(str(run_dir.resolve()))


def setup_state(tc):
    tc.state = TrainingState()
    tc.accelerator.register_for_checkpointing(tc.state)
    load_state(tc.accelerator, tc.state_path)


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
        if tc.accelerator.is_main_process and hasattr(tc, 'sparsity'):
            tc.writer.add_scalar('Eval/Test sparsity', tc.sparsity, global_step=tc.state.current_batch)
        train_loss, train_acc = test_classification(tc.accelerator,
                                                    tc.model,
                                                    tc.train_eval_loader,
                                                    tc.criterion_type,
                                                    batches=args.eval_batches)
        if tc.accelerator.is_main_process and hasattr(tc, 'sparsity'):
            tc.writer.add_scalar('Eval/Train sparsity', tc.sparsity, global_step=tc.state.current_batch)
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Eval/Test loss', test_loss, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Eval/Test accuracy', test_acc, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Eval/Train loss', train_loss, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Eval/Train accuracy', train_acc, global_step=tc.state.current_batch)


def training_loop(args, tc):
    # TODO reduce code duplication - see code in `methods`
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
        running_loss, running_distill_loss = 0, 0
        for _ in range(args.gradient_accumulation_steps):
            # forward
            try:
                X, y = next(train_iter)
            except StopIteration:
                train_iter = iter(tc.train_loader)
                X, y = next(train_iter)
            if mixup_fn is not None:
                X, y = mixup_fn(X, y)
            y_pred = tc.model(X)
            # loss computation
            if len(y_pred.shape) == 3:
                # For CE loss on sequences
                y_pred = y_pred.transpose(1, 2)
            loss = tc.criterion(y_pred, y)
            if args.distill_from is not None:
                y_teacher_logprobs = torch.log_softmax(tc.teacher_model(X), dim=-1)
                distill_loss = tc.distill_criterion(torch.log_softmax(y_pred, dim=-1), y_teacher_logprobs.detach())
                loss = loss + tc.distill_weight * distill_loss
            # rescale loss if doing gradient accumulation
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                if args.distill_from is not None:
                    distill_loss = distill_loss / args.gradient_accumulation_steps
            # backward the loss
            tc.accelerator.backward(loss)
            running_loss += loss.item()
            if args.distill_from is not None:
                running_distill_loss += distill_loss.item()
        if args.clip_grad_norm is not None:
            total_norm = tc.accelerator.clip_grad_norm_(tc.model.parameters(), args.clip_grad_norm)
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/Gradient norm', total_norm.item(), global_step=tc.state.current_batch)
        # update parameters
        tc.optimizer.step()
        if tc.scheduler is not None:
            # log LRs
            if tc.accelerator.is_main_process:
                for i, lr in enumerate(get_lrs(tc.optimizer)):
                    tc.writer.add_scalar(f'Train/Group {i} LR', lr, global_step=tc.state.current_batch)
            if args.scheduler_class == 'reduce_on_plateau':
                tc.scheduler.step(running_loss)
            else:
                tc.scheduler.step()
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Train/Loss', running_loss, global_step=tc.state.current_batch)
            if args.distill_from is not None:
                tc.writer.add_scalar('Train/Distill loss', running_distill_loss, global_step=tc.state.current_batch)
        # bookkeeping
        tc.state.current_batch += 1


def final_eval(args, tc):
    if not tc.final_path.exists():
        if tc.accelerator.is_main_process:
            save_state(tc.accelerator, tc.state_path)
        # test on testset
        test_loss, test_acc = test_classification(tc.accelerator, tc.model, tc.test_loader, tc.criterion_type,
                                                  args.test_batches)
        if tc.accelerator.is_main_process:
            final_results = {}
            final_results['args'] = args
            unwrapped_model = tc.accelerator.unwrap_model(tc.model)
            final_results['model_state'] = unwrapped_model.state_dict()
            tc.writer.add_scalar('Eval/Test loss', test_loss, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Eval/Test accuracy', test_acc, global_step=tc.state.current_batch)
            final_results['final_score'] = test_acc
            final_results['final_loss'] = test_loss
            logging.info(f'Test accuracy: {test_acc}')
            # benchmark model efficiency
            model_costs, model_params = benchmark(unwrapped_model, tc.test_loader)
            final_results['model_flops'] = model_costs.total()
            final_results['model_flops_by_module'] = dict(model_costs.by_module())
            final_results['model_flops_by_operator'] = dict(model_costs.by_operator())
            final_results['model_params'] = dict(model_params)
            tc.writer.add_scalar('Eval/Model FLOPs', model_costs.total(), global_step=tc.state.current_batch)
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
