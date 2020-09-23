# encoding: utf-8
"""
@author: ccj
@contact:
"""

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20,
                                     val_shift_limit=0, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=(0.5, 1.5), p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5)
            ], p=0.5),
            A.CoarseDropout(max_holes=8, max_height=16,
                            max_width=16, p=0.5),
            A.Resize(height=512, width=512, p=1.),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.),
            ToTensorV2(p=1.),
        ]
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1.),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.),
            ToTensorV2(p=1.),
        ]
    )


def get_test_transform():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.),
            ToTensorV2(p=1.),
        ]
    )


if __name__ == "__main__":
    train_aug = get_train_transforms()
    valid_aug = get_valid_transforms()
    test_aug = get_test_transform()
    A.save(train_aug, 'train_aug.yaml', data_format='yaml')
    A.save(valid_aug, 'valid_aug.yaml', data_format='yaml')
    A.save(test_aug, 'test_aug.yaml', data_format='yaml')
