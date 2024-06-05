import warnings
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandAugment,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from contrastors.config import AugmentationCfg
from contrastors.dataset.constants import OPENAI_IMAGE_DATASET_MEAN, OPENAI_IMAGE_DATASET_STD


class ResizeMaxSize(nn.Module):
    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(
    image_size: int,
    is_train: bool,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    resize_longest_max: bool = False,
    fill_color: int = 0,
    aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    mean = mean or OPENAI_IMAGE_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_IMAGE_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()

    normalize = Normalize(mean=mean, std=std)
    if is_train:
        aug_cfg_dict = {k: v for k, v in aug_cfg.dict().items() if v is not None}
        train_transform = Compose(
            [
                # RandAugment(),
                RandomResizedCrop(
                    image_size,
                    scale=aug_cfg_dict.pop('scale'),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                # RandomHorizontalFlip(),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )
        if aug_cfg_dict:
            warnings.warn(f'Unused augmentation cfg items, specify `use_timm` to use ({list(aug_cfg_dict.keys())}).')
        return train_transform
    else:
        if resize_longest_max:
            transforms = [ResizeMaxSize(image_size, fill=fill_color)]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend(
            [
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )
        return Compose(transforms)
