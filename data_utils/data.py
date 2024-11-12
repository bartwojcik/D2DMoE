import math
import os
import random
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import ImageFilter, ImageOps
from datasets import load_dataset, load_from_disk
from sklearn.datasets import fetch_20newsgroups
from torchvision import datasets, transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_V2_S_Weights,
    Swin_V2_S_Weights,
    ViT_B_16_Weights,
)
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer

DOWNLOAD = False


def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    dataset_path = os.environ['MNIST_PATH']
    train_data = datasets.MNIST(dataset_path, train=True, download=DOWNLOAD, transform=transform)
    train_eval_data = train_data
    test_data = datasets.MNIST(dataset_path, train=False, download=DOWNLOAD, transform=transform)
    return train_data, train_eval_data, test_data


def get_cifar10(normalization=None):
    if normalization == '0.5':
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean, std)
    elif normalization == 'skip':
        normalize = transforms.Compose([])
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
        normalize = transforms.Normalize(mean, std)
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    transform_train = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.1),
    ])
    dataset_path = os.environ['CIFAR10_PATH']
    train_data = datasets.CIFAR10(dataset_path, train=True, download=DOWNLOAD, transform=transform_train)
    train_eval_data = datasets.CIFAR10(dataset_path, train=True, download=DOWNLOAD, transform=transform_eval)
    test_data = datasets.CIFAR10(dataset_path, train=False, download=DOWNLOAD, transform=transform_eval)
    return train_data, train_eval_data, test_data


def get_cifar100(normalization=True):
    if normalization == '0.5':
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean, std)
    elif normalization == 'skip':
        normalize = transforms.Compose([])
    else:
        mean, std = (0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276)
        normalize = transforms.Normalize(mean, std)
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    transform_train = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.1),
    ])
    dataset_path = os.environ['CIFAR100_PATH']
    train_data = datasets.CIFAR100(dataset_path, train=True, download=DOWNLOAD, transform=transform_train)
    train_eval_data = datasets.CIFAR100(dataset_path, train=True, download=DOWNLOAD, transform=transform_eval)
    test_data = datasets.CIFAR100(dataset_path, train=False, download=DOWNLOAD, transform=transform_eval)
    return train_data, train_eval_data, test_data


class GaussianBlur:
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


class Solarization:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class GrayScale:
    def __init__(self, p=0.2):
        self.p = p
        self.transform = transforms.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transform(img)
        else:
            return img


class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BICUBIC):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]
        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return transforms.functional.resized_crop(img, i, j, h, w, self.size, interpolation)


def get_imagenet(normalization=None, variant=None, image_size=None):
    if normalization == '0.5':
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean, std)
    elif normalization == 'skip':
        normalize = transforms.Compose([])
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean, std)
    img_size = 224 if image_size is None else image_size
    if variant == 'trivial_augment':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.TrivialAugmentWide(interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1),
        ])
        transform_eval = transforms.Compose([
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])
    elif variant == 'deit3_rrc':
        # based on https://arxiv.org/pdf/2204.07118.pdf
        # https://github.com/facebookresearch/deit/blob/main/augment.py
        transform_train = transforms.Compose([
            RandomResizedCropAndInterpolation(img_size,
                                              scale=(0.08, 1.0),
                                              interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([GrayScale(p=1.0),
                                     Solarization(p=1.0),
                                     GaussianBlur(p=1.0)]),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            normalize,
        ])
        transform_eval = transforms.Compose([
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])
    elif variant is None or 'deit3' in variant:
        # based on https://arxiv.org/pdf/2204.07118.pdf
        # https://github.com/facebookresearch/deit/blob/main/augment.py
        transform_train = transforms.Compose([
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([GrayScale(p=1.0),
                                     Solarization(p=1.0),
                                     GaussianBlur(p=1.0)]),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            normalize,
        ])
        transform_eval = transforms.Compose([
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])
    elif variant == 'tv_convnext_t':
        transform_train = transforms.Compose([
            transforms.Resize(236, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.RandAugment(interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            normalize,
        ])
        transform_eval = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
    elif variant == 'tv_efficientnet_v2_s':
        transform_train = transforms.Compose([
            transforms.Resize(384, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(384),
            transforms.RandAugment(interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            normalize,
        ])
        transform_eval = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
    elif variant == 'tv_efficientnet_b0':
        transform_train = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.AutoAugment(interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        transform_eval = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
    elif variant == 'tv_vit_b_16':
        transform_train = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.RandAugment(interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1),
        ])
        transform_eval = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    elif variant == 'tv_swin_v2_s':
        transform_train = transforms.Compose([
            transforms.Resize(260, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.RandAugment(interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1),
        ])
        transform_eval = Swin_V2_S_Weights.IMAGENET1K_V1.transforms()
    dataset_path = os.environ['IMAGENET_PATH']
    train_data = datasets.ImageNet(dataset_path, transform=transform_train)
    train_eval_data = datasets.ImageNet(dataset_path, transform=transform_eval)
    test_data = datasets.ImageNet(dataset_path, split='val', transform=transform_eval)
    return train_data, train_eval_data, test_data


def get_imagenet_c(severity, distortion=None, normalization=None, variant=None, image_size=None):
    if normalization == '0.5':
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean, std)
    elif normalization == 'skip':
        normalize = transforms.Compose([])
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean, std)
    img_size = 224 if image_size is None else image_size
    if distortion:
        dataset_path = str(Path(os.environ['IMAGENET_C_PATH']) / str(distortion) / str(severity))
    else:
        dataset_path = str(Path(os.environ['IMAGENET_C_AGGREGATED_PATH']) / str(severity))
    if variant == 'trivial_augment':
        transform_eval = transforms.Compose([
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])
    elif variant == 'deit3_rrc':
        transform_eval = transforms.Compose([
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])
    elif variant is None or 'deit3' in variant:
        transform_eval = transforms.Compose([
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])
    elif variant == 'tv_convnext_t':
        transform_eval = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
    elif variant == 'tv_efficientnet_v2_s':
        transform_eval = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
    elif variant == 'tv_efficientnet_b0':
        transform_eval = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
    elif variant == 'tv_vit_b_16':
        transform_eval = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    elif variant == 'tv_swin_v2_s':
        transform_eval = Swin_V2_S_Weights.IMAGENET1K_V1.transforms()
    test_data = datasets.ImageFolder(dataset_path, transform=transform_eval)
    return None, None, test_data


def get_tinyimagenet(normalization=None, variant=None):
    if normalization == '0.5':
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean, std)
    elif normalization == 'skip':
        normalize = transforms.Compose([])
    else:
        mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
        normalize = transforms.Normalize(mean, std)
    if variant is None or variant == 'trivial_augment':
        transform_train = transforms.Compose([
            transforms.TrivialAugmentWide(),
            transforms.RandomCrop(64, padding=8),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1),
        ])
    elif variant == 'deit3':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([GrayScale(p=0.9),
                                     Solarization(p=0.9),
                                     GaussianBlur(p=0.9)]),
            transforms.ToTensor(),
            normalize,
        ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset_path = os.environ['TINYIMAGENET_PATH']
    train_path = f'{dataset_path}/train'
    # TODO include the script that generates the symlinks somewhere
    # test_path = f'{dataset_path}/val'
    test_path = f'{dataset_path}/val/images'
    train_data = datasets.ImageFolder(train_path, transform=transform_train)
    train_eval_data = datasets.ImageFolder(train_path, transform=transform_eval)
    test_data = datasets.ImageFolder(test_path, transform=transform_eval)
    return train_data, train_eval_data, test_data


def get_cubbirds(normalization=None, variant=None, image_size=None):
    if normalization == '0.5':
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean, std)
    elif normalization == 'skip':
        normalize = transforms.Compose([])
    else:
        raise NotImplementedError()
    img_size = 224 if image_size is None else image_size
    if variant is None or variant == 'trivial_augment':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=InterpolationMode.BILINEAR),
            transforms.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1),
        ])
    elif variant == 'deit3':
        # based on https://arxiv.org/pdf/2204.07118.pdf
        # https://github.com/facebookresearch/deit/blob/main/augment.py
        transform_train = transforms.Compose([
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([GrayScale(p=0.9),
                                     Solarization(p=0.9),
                                     GaussianBlur(p=0.9)]),
            transforms.ToTensor(),
            normalize,
        ])
    transform_eval = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    dataset_path = os.environ['CUBBIRDS_PATH']
    # TODO include the script that generates the symlinks somewhere
    trainset_path = f'{dataset_path}/images_train_test/train'
    eval_path = f'{dataset_path}/images_train_test/val'
    train_data = datasets.ImageFolder(trainset_path, transform=transform_train)
    train_eval_data = datasets.ImageFolder(trainset_path, transform=transform_eval)
    test_data = datasets.ImageFolder(eval_path, transform=transform_eval)
    return train_data, train_eval_data, test_data


def get_food101(normalization=None, variant=None, image_size=None):
    if normalization == '0.5':
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean, std)
    elif normalization == 'skip':
        normalize = transforms.Compose([])
    else:
        raise NotImplementedError()
    img_size = 224 if image_size is None else image_size
    if variant is None or variant == 'trivial_augment':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=InterpolationMode.BILINEAR),
            transforms.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1),
        ])
    elif variant == 'deit3':
        # based on https://arxiv.org/pdf/2204.07118.pdf
        # https://github.com/facebookresearch/deit/blob/main/augment.py
        transform_train = transforms.Compose([
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([GrayScale(p=0.9),
                                     Solarization(p=0.9),
                                     GaussianBlur(p=0.9)]),
            transforms.ToTensor(),
            normalize,
        ])
    transform_eval = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    dataset_path = os.environ['FOOD101_PATH']
    train_data = datasets.Food101(dataset_path, split='train', transform=transform_train)
    train_eval_data = datasets.Food101(dataset_path, split='train', transform=transform_eval)
    test_data = datasets.Food101(dataset_path, split='test', transform=transform_eval)
    return train_data, train_eval_data, test_data


class NlpHfDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        y = sample.pop('label')
        return sample, y


class NlpDictDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset['label'])

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self.dataset.items()}
        y = sample.pop('label')
        return sample, y


def get_rte(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    return get_glue_dataset('rte', tokenizer_name=tokenizer_name, padding=padding, max_seq_length=max_seq_length,
                            truncation=truncation)


def get_qqp(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    return get_glue_dataset('qqp', tokenizer_name=tokenizer_name, padding=padding, max_seq_length=max_seq_length,
                            truncation=truncation)


def get_qnli(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    return get_glue_dataset('qnli', tokenizer_name=tokenizer_name, padding=padding, max_seq_length=max_seq_length,
                            truncation=truncation)


def get_mrpc(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    return get_glue_dataset('mrpc', tokenizer_name=tokenizer_name, padding=padding, max_seq_length=max_seq_length,
                            truncation=truncation)


def get_sst2(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    return get_glue_dataset('sst2', tokenizer_name=tokenizer_name, padding=padding, max_seq_length=max_seq_length,
                            truncation=truncation)


def get_glue_dataset(task_name, tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    train_dataset = load_dataset("glue", task_name, split='train', cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])
    validation_dataset = load_dataset("glue", task_name, split='validation',
                                      cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    task_to_keys = {
        # "cola": ("sentence", None),
        # "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        # "stsb": ("sentence1", "sentence2"),
        # "wnli": ("sentence1", "sentence2"),
    }
    if task_name not in task_to_keys.keys():
        raise NotImplementedError()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              use_fast=True,
                                              cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])
    sentence1_key, sentence2_key = task_to_keys[task_name]

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=truncation
        )

        return result

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc=f"Running tokenizer on {task_name}:train",
    )
    validation_dataset = validation_dataset.map(
        preprocess_function,
        batched=True,
        desc=f"Running tokenizer on {task_name}:validation",
    )

    if tokenizer_name.startswith('bert'):
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
        validation_dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
    elif 'roberta' in tokenizer_name or 'distilbert' in tokenizer_name:
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
        validation_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
    else:
        raise NotImplementedError()

    # GLUE tasks are tested on validation set
    return NlpHfDatasetWrapper(train_dataset), \
        NlpHfDatasetWrapper(validation_dataset), \
        NlpHfDatasetWrapper(validation_dataset)


def get_ag_news(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    train_dataset = load_dataset("ag_news", split='train', cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])
    test_dataset = load_dataset("ag_news", split='test', cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              use_fast=True,
                                              cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    def preprocess_function(examples):

        result = tokenizer(
            examples['text'], padding=padding, max_length=max_seq_length, truncation=truncation
        )

        return result

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on ag_news:train",
    )
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on ag_news:test",
    )

    if tokenizer_name.startswith('bert'):
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
    elif 'roberta' in tokenizer_name or 'distilbert' in tokenizer_name:
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
    else:
        raise NotImplementedError()

    # GLUE tasks are tested on validation set
    return NlpHfDatasetWrapper(train_dataset), \
        NlpHfDatasetWrapper(test_dataset), \
        NlpHfDatasetWrapper(test_dataset)


def get_dbpedia14(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    train_dataset = load_dataset("dbpedia_14", split='train', cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])
    test_dataset = load_dataset("dbpedia_14", split='test', cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              use_fast=True,
                                              cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    def preprocess_function(examples):

        result = tokenizer(
            examples['content'], padding=padding, max_length=max_seq_length, truncation=truncation
        )

        return result

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dbpedia14:train",
    )
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dbpedia14:test",
    )

    if tokenizer_name.startswith('bert'):
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
    elif 'roberta' in tokenizer_name or 'distilbert' in tokenizer_name:
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
    else:
        raise NotImplementedError()

    # GLUE tasks are tested on validation set
    return NlpHfDatasetWrapper(train_dataset), \
        NlpHfDatasetWrapper(test_dataset), \
        NlpHfDatasetWrapper(test_dataset)


def get_20newsgroups(tokenizer_name=None, padding="max_length", max_seq_length=512, truncation=True):
    newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, data_home=os.environ['TRANSFORMERS_CACHE_DIR'])
    newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, data_home=os.environ['TRANSFORMERS_CACHE_DIR'])

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              use_fast=True,
                                              cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    is_bert = tokenizer_name.startswith('bert')

    def preprocess_function(examples):
        result = tokenizer(
            examples.data,
            add_special_tokens=True,
            return_token_type_ids=is_bert,
            padding=padding,
            max_length=max_seq_length,
            truncation=truncation
        )

        output = {
            'input_ids': torch.tensor(result['input_ids']),
            'attention_mask': torch.tensor(result['attention_mask']),
            'label': torch.tensor(examples.target)
        }
        if is_bert:
            output['token_type_ids'] = torch.tensor(result['token_type_ids'])
        return output

    train_dataset = preprocess_function(newsgroups_train)
    validation_dataset = preprocess_function(newsgroups_test)

    return NlpDictDatasetWrapper(train_dataset), \
        NlpDictDatasetWrapper(validation_dataset), \
        NlpDictDatasetWrapper(validation_dataset)


def get_emotion(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    train_dataset = load_dataset("dair-ai/emotion", split="train", cache_dir=os.environ["TRANSFORMERS_CACHE_DIR"])
    validation_dataset = load_dataset(
        "dair-ai/emotion", split="validation", cache_dir=os.environ["TRANSFORMERS_CACHE_DIR"]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True, cache_dir=os.environ["TRANSFORMERS_CACHE_DIR"]
    )

    def preprocess_function(examples):
        result = tokenizer(examples["text"], padding=padding, max_length=max_seq_length, truncation=truncation)
        return result

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on emotion:train",
    )
    validation_dataset = validation_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on emotion:validation",
    )

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"],
    )
    validation_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"],
    )

    return (
        NlpHfDatasetWrapper(train_dataset),
        NlpHfDatasetWrapper(train_dataset),
        NlpHfDatasetWrapper(validation_dataset),
    )


class GPTDatasetWrapper(torch.utils.data.Dataset):
    """
    Wrapper for text datasets for next token prediction. This is a bit tricky because datasets usually map
    idx to the list of token ids for subsequent texts, but our inputs and targets are not texts but tokens shifted by 1.
    So the dataset has to take this into account - the x's and y's are not text encodings, but token sequences.
    """

    def __init__(self, dataset_path, sequence_length: int):
        super().__init__()
        self.data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
        self.sequence_length = sequence_length

    def __len__(self):
        return 100000  # TODO we always sample at random so shouldn't matter

    def __getitem__(self, idx: int):
        ix = torch.randint(len(self.data) - self.sequence_length, (1,))
        x = torch.from_numpy((self.data[ix:ix + self.sequence_length]).astype(np.int64))
        y = torch.from_numpy((self.data[ix + 1:ix + 1 + self.sequence_length]).astype(np.int64))
        return x, y


class C4DatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset, tokenizer, seq_length):
        self.dataset = dataset
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        # set model max length to high number to disable warnings
        # we handle sequence length ourselves
        self.tokenizer.model_max_length = 100_000
        if "llama" in tokenizer.name_or_path.lower():
            self.tokenizer.pad_token = self.tokenizer.eos_token  # TODO llama does not have a padding token
            self.sequence_separator_id = self.tokenizer.bos_token_id  # TODO llama does not have a separator token
        elif 'gemma' in tokenizer.name_or_path.lower():
            """Gemma has a padding token already, but does not have a sentence separator"""
            self.sequence_separator_id = self.tokenizer.bos_token_id
        else:
            raise NotImplementedError("Only Llama tokenizer supported")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        tokenized = self.get_tokenized_sample(idx)
        tokens = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        target_tokens = tokens[1:] + [self.tokenizer.pad_token_id]
        inputs = {'input_ids': tokens, 'attention_mask': attention_mask}
        inputs = {k: torch.tensor(v) for k, v in inputs.items()}
        labels = torch.tensor(target_tokens)
        return inputs, labels

    def get_tokenized_sample(self, idx):
        """
        Sample examples from the dataset until we reach the desired sequence length.
        """
        result = dict()
        result["input_ids"] = []
        result["attention_mask"] = []
        current_length = 0
        document_id = idx
        rand = random.Random(idx)
        while current_length < self.seq_length:
            example = self.get_one_example(
                document_id, self.seq_length - current_length
            )
            current_length += len(example)
            result["input_ids"] += example
            result["attention_mask"] += [1] * len(example)
            if current_length < self.seq_length:
                result["input_ids"] += [self.sequence_separator_id]
                result["attention_mask"] += [0]
                current_length += 1
                document_id = rand.randint(0, len(self.dataset) - 1)
        return result

    def get_one_example(self, idx, length):
        text = self.dataset[idx]["text"]
        ids = self.tokenizer(text)["input_ids"]
        if len(ids) > length:
            return self.get_random_chunk(ids, length)
        else:
            return ids

    def get_random_chunk(self, ids, length):
        beginning = random.randint(0, len(ids) - length)
        return ids[beginning: beginning + length]


def get_c4(seq_length=1024, cache_dir=None, tokenizer_name=None, dataset_name=None, train_dataset_dir=None,
           val_dataset_dir=None):
    assert tokenizer_name is not None, 'Please provide a tokenizer name for C4 dataset.'
    assert dataset_name is not None or (train_dataset_dir is not None and val_dataset_dir is not None), \
        'Please provide a dataset name or train and validation dataset directories.'
    if cache_dir is None:
        cache_dir = Path(os.environ['C4_DATA_DIR'])
        cache_dir.mkdir(parents=True, exist_ok=True)
    if train_dataset_dir is not None:
        # '/raid/NFS_SHARE/datasets/c4/train/c4_train'
        train = load_from_disk(train_dataset_dir)
    else:
        train = load_dataset(dataset_name, split="train", name='en', cache_dir=cache_dir)
    if val_dataset_dir is not None:
        # '/raid/NFS_SHARE/datasets/c4/validation/c4_validation'
        val = load_from_disk(val_dataset_dir)
    else:
        if dataset_name == "stas/c4-en-10k":
            val = train
        else:
            val = load_dataset(dataset_name, split="validation", name='en', cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        # Some tokenizers have no padding token which causes bugs
        tokenizer.pad_token = tokenizer.eos_token
    train = C4DatasetWrapper(train, tokenizer, seq_length)
    val = C4DatasetWrapper(val, tokenizer, seq_length)
    return train, train, val


def get_openwebtext(sequence_length=1024):
    dataset_dir = Path(os.environ['GPT_DATA_DIR']) / 'openwebtext'

    ds_train = GPTDatasetWrapper(dataset_dir / 'train.bin', sequence_length=sequence_length)
    ds_val = GPTDatasetWrapper(dataset_dir / 'val.bin', sequence_length=sequence_length)
    return ds_train, ds_train, ds_val


def get_shakespeare_char(sequence_length=1024):
    dataset_dir = Path(os.environ['GPT_DATA_DIR']) / 'shakespeare_char'

    ds_train = GPTDatasetWrapper(dataset_dir / 'train.bin', sequence_length=sequence_length)
    ds_val = GPTDatasetWrapper(dataset_dir / 'val.bin', sequence_length=sequence_length)
    return ds_train, ds_train, ds_val


DATASETS_NAME_MAP = {
    'mnist': get_mnist,
    'cifar10': get_cifar10,
    'cifar100': get_cifar100,
    'tinyimagenet': get_tinyimagenet,
    'imagenet': get_imagenet,
    'imagenet-c-1': partial(get_imagenet_c, 1),
    'imagenet-c-2': partial(get_imagenet_c, 2),
    'imagenet-c-3': partial(get_imagenet_c, 3),
    'imagenet-c-4': partial(get_imagenet_c, 4),
    'imagenet-c-5': partial(get_imagenet_c, 5),
    'cubbirds': get_cubbirds,
    'food101': get_food101,
    'rte': get_rte,
    'qqp': get_qqp,
    'qnli': get_qnli,
    'mrpc': get_mrpc,
    'sst2': get_sst2,
    '20newsgroups': get_20newsgroups,
    'ag_news': get_ag_news,
    'dbpedia14': get_dbpedia14,
    'openwebtext': get_openwebtext,
    'shakespeare_char': get_shakespeare_char,
    'emotion': get_emotion,
    'c4': get_c4,
}

DATASET_TO_SEQUENCE_LENGTH = {
    'rte': 128,
    'qqp': 128,
    'qnli': 128,
    'mrpc': 128,
    'sst2': 128,
    '20newsgroups': 512,
    'ag_news': 128,
    'dbpedia14': 128,
    'emotion': 128,
}
DATASET_TO_NUM_CLASSES = {
    'rte': 2,
    'qqp': 2,
    'qnli': 2,
    'mrpc': 2,
    'sst2': 2,
    '20newsgroups': 20,
    'ag_news': 4,
    'dbpedia14': 14,
    'emotion': 6,
}
