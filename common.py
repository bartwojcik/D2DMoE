import math

import omegaconf
import torch.nn
from torch import nn, optim
from torch.nn.init import calculate_gain
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_inverse_sqrt_schedule,
    get_linear_schedule_with_warmup,
)

from architectures.avit import AvitWrapper
from architectures.early_exits.gpf import GPF
from architectures.early_exits.l2w import L2W
from architectures.early_exits.pbee import PBEE
from architectures.early_exits.sdn import SDN
from architectures.early_exits.ztw import ZTWCascading, ZTWEnsembling
from architectures.efficientnet import EfficientNetV2
from architectures.nlp import get_bert, get_distilbert, get_gpt2, get_roberta, get_gemma
from architectures.pretrained import (
    get_convnext_t,
    get_efficientnet_b0,
    get_efficientnet_v2_s,
    get_swin_v2_s,
    get_vit_b_16,
)
from architectures.resnets import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from architectures.vgg import VGG16BN
from architectures.vit import VisionTransformer


def default_init(model):
    for m in model.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()


def vgg_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def get_default_args():
    default_args = omegaconf.OmegaConf.create()

    default_args.exp_id = 0  # experiment id
    default_args.runs_dir = "runs"  # directory to save the results to
    default_args.model_class = None  # class of the model to train
    default_args.model_args = None  # arguments to be passed to the model init function
    default_args.dataset = None  # dataset to train on
    default_args.dataset_args = None  # customization arguments for the dataset
    default_args.mixup_alpha = None  # alpha parameter for mixup's beta distribution
    default_args.cutmix_alpha = None  # alpha parameter for cutmix's beta distribution
    default_args.mixup_mode = None  # how to apply mixup/cutmix ('batch', 'pair' or 'elem')
    default_args.mixup_smoothing = None  # label smoothing when using mixup
    default_args.init_fun = None  # parameters init function to be used
    default_args.batch_size = None  # batch size for training
    default_args.test_batch_size = None  # batch size for evaluation
    default_args.loss_type = None  # loss function to be used for training
    default_args.loss_args = None  # arguments to be passed to the loss init function
    default_args.optimizer_class = None  # class of the optimizer to use for training
    default_args.optimizer_args = None  # arguments to be passed to the optimizer init function
    default_args.scheduler_class = None  # class of the scheduler to use for training
    default_args.scheduler_args = None  # arguments to be passed to the scheduler init function
    default_args.clip_grad_norm = None  # gradient clipping norm
    default_args.epochs = None  # number of epochs to train for
    default_args.last_batch = None  # number of iterations to train for; use either this or epochs
    default_args.gradient_accumulation_steps = 1  # number of gradient accumulation steps
    default_args.mixed_precision = None  # whether to use accelerate's mixed precision
    default_args.num_workers = 8  # number of workers to use for data loading
    default_args.eval_points = 100  # number of short evaluations on the validation/test data while training
    default_args.eval_batches = 10  # number of batches to evaluate on each time while training
    default_args.test_batches = 0  # number of batches for final test phase; 0 means full dataset
    default_args.save_every = 10  # save model every N minutes
    default_args.use_wandb = False  # use weights and biases
    default_args.cpu = False  # Train on CPU for local debugging
    default_args.save_test_activations = False  # Save activations from test set for ablations

    # use only None for method specific args, and fill them with default values in the code!
    # unless they are not used in generate_run_name(), changing the defaults changes run names for unrelated runs!
    # method specific args
    default_args.base_on = None  # unique experiment name to use the model from
    default_args.distill_from = None # unique experiment name to use the model from for distillation
    default_args.distill_weight = None

    # Early Exit specific
    default_args.with_backbone = None  # whether to train the backbone network along with the heads
    default_args.auxiliary_loss_type = None  # type of the auxiliary loss for early-exit heads
    default_args.auxiliary_loss_weight = None  # weight of the auxiliary loss for early-exit heads
    default_args.eval_thresholds = None  # number of early exit thresholds to evaluate
    # L2W
    default_args.l2w_meta_interval = None  # every n-th batch will have meta step enabled
    default_args.wpn_width = None  # weight prediction network hidden layers width
    default_args.wpn_depth = None  # weight prediction network depth
    default_args.wpn_optimizer_class = None  # class of the optimizer to use for training the weight prediction network
    default_args.wpn_optimizer_args = None  # arguments to be passed to the WPN optimizer init function
    default_args.wpn_scheduler_class = None  # class of the scheduler to use for training the weight prediction network
    default_args.wpn_scheduler_args = None  # arguments to be passed to the WPN scheduler init function
    default_args.l2w_epsilon = None  # epsilon which weights the pseudo weights (0.3 used in the paper)
    default_args.l2w_target_p = None  # target p that determines the head budget allocation (15 used in the paper)

    # MoE specific
    default_args.importance_loss_weight = None  # weight of the importance loss in top k routing
    default_args.load_balancing_loss_weight = None  # weight of the load balancing loss in top k routing
    # MoEfication specific
    default_args.k_to_eval = None  # list of ks to evaluate moefication model with
    default_args.router_loss_type = None  # loss function to be used for training the routers
    default_args.router_loss_args = None  # arguments to be passed to the router loss init function
    # DSTI specific
    default_args.dsti_distill_loss_type = None  # loss type for representation distillation stage
    default_args.dsti_enforce_mode = None  # activation sparsity enforcement mode
    default_args.dsti_enforce_weight = None  # activation sparsity enforcement weight
    default_args.dsti_enforce_schedule = None  # activation sparsity enforcement weight
    default_args.dsti_clamp_displacement = None  # activation sparsity enforcement weight
    default_args.dsti_tau_to_eval = None  # tau threshold for selecting experts to execute
    default_args.dsti_router_labels_layer = None  # layer to use for labels construction for router training
    default_args.dsti_router_labels_norm = None  # torch.linalg.vector_norm's ord to use for labels construction for router training
    default_args.dsti_separate_router_training = None  # trains the router sequentially instead of simultaneously to save GPU memory
    default_args.dsti_quantization = None  # quantization scheme to use; None defaults to no quantization


    # A-ViT specific
    default_args.ponder_loss_weight = None  # weight of the pondering loss
    default_args.distr_target_depth = None  # mean depth of the distributional prior
    default_args.distr_prior_loss_weight = None  # weight of the distributional prior regularization

    return default_args


INIT_NAME_MAP = {
    'default': default_init,
    'vgg': vgg_init,
    None: None,
}


def function_to_module(torch_function):
    class ModularizedFunction(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch_function(x)

    return ModularizedFunction


ACTIVATION_NAME_MAP = {
    'relu': torch.nn.ReLU,
    'gelu': torch.nn.GELU,
    'sigmoid': torch.nn.Sigmoid,
    'tanh': torch.nn.Tanh,
    'leaky_relu': torch.nn.LeakyReLU,
    'softplus': torch.nn.Softplus,
    'silu': torch.nn.SiLU,
    'abs': function_to_module(torch.abs),
    'identity': torch.nn.Identity,
}

LOSS_NAME_MAP = {
    'ce': nn.CrossEntropyLoss,
    'bcewl': nn.BCEWithLogitsLoss,
    'bce': nn.BCELoss,
    'nll': nn.NLLLoss,
    'mse': nn.MSELoss,
    'mae': nn.L1Loss,
    'huber': nn.HuberLoss,
}

OPTIMIZER_NAME_MAP = {
    'sgd': optim.SGD,
    'adam': optim.AdamW,
    'adagrad': optim.Adagrad,
}

SCHEDULER_NAME_MAP = {
    'reduce_on_plateau': optim.lr_scheduler.ReduceLROnPlateau,
    'cosine': optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warm_restarts': optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'linear': get_linear_schedule_with_warmup,
    'cosine_with_warmup': get_cosine_schedule_with_warmup,
    'inverse_sqrt': get_inverse_sqrt_schedule,
}

MODEL_NAME_MAP = {
    'vgg16bn': VGG16BN,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'efficientnetv2': EfficientNetV2,
    'vit': VisionTransformer,
    'avit': AvitWrapper,
    'sdn': SDN,
    'pbee': PBEE,
    'gpf': GPF,
    'l2w': L2W,
    'ztw_cascading': ZTWCascading,
    'ztw_ensembling': ZTWEnsembling,
    'tv_convnext_t': get_convnext_t,
    'tv_efficientnet_b0': get_efficientnet_b0,
    'tv_efficientnet_s': get_efficientnet_v2_s,
    'tv_vit_b_16': get_vit_b_16,
    'tv_swin_v2_s': get_swin_v2_s,
    'bert': get_bert,
    'distilbert': get_distilbert,
    'roberta': get_roberta,
    'gpt2': get_gpt2,
    'gemma': get_gemma
}
