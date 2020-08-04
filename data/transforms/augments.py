# encoding: utf-8
"""
@author: ccj
@contact:
"""

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(cfg):
    return A.Compose(
        [
            A.HorizontalFlip(p=cfg.hflip_prob),
            A.VerticalFlip(p=cfg.vflip_prob),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=cfg.hue_limit, sat_shift_limit=cfg.sat_limit,
                                     val_shift_limit=cfg.val_limit, p=cfg.hue_prob),
                A.RandomBrightnessContrast(brightness_limit=cfg.brightness_limit,
                                           contrast_limit=cfg.contrast_limit, p=cfg.contrast_prob),
            ], p=0.5),
            A.CoarseDropout(max_holes=cfg.coarse_max_holes, max_height=cfg.coarse_max_height,
                            max_width=cfg.coarse_max_width, p=cfg.coarse_prob),
            # A.Resize(height=512, width=512, p=1.),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(p=1.),
        ]
    )


def get_valid_transforms(cfg):
    return A.Compose(
        [
            A.HorizontalFlip(p=cfg.hflip_prob),
            A.VerticalFlip(p=cfg.vflip_prob),
            # A.Resize(height=512, width=512, p=1.),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(p=1.),
        ]
    )


def get_test_transform(cfg):
    return A.Compose(
        [
            # A.Resize(height=512, width=512, p=1.),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(p=1.),
        ]
    )
