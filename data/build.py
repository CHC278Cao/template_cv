# encoding: utf-8
"""
@author: ccj
@contact:
"""

import os
import pdb
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import DataLoader

from .datasets.utils import *
from .datasets.train_datasets import TrainDataset
from .datasets.test_datasets import TestDataset
from .transforms.augments import get_train_transforms, get_valid_transforms, get_test_transform
from .transforms.utils import MixCollator

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    print("TPU is not used")


def split_dataset(cfg):
    df = pd.read_csv(cfg.train_file)
    df = df.sample(frac=1., random_state=cfg.seed).reset_index(drop=True)

    df["classes"] = df["healthy"] * 0 + df["multiple_diseases"] * 1 \
                    + df["rust"] * 2 + df["scab"] * 3
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for idx, (train_idx, valid_idx) in enumerate(skf.split(X=df, y=df["classes"].values)):
        df.loc[valid_idx, "fold"] = idx

    train_df = df[df["fold"] != cfg.fold_idx].reset_index(drop=True)
    valid_df = df[df["fold"] == cfg.fold_idx].reset_index(drop=True)

    return df, train_df, valid_df


def build_dataset(cfg):
    df, train_df, valid_df = split_dataset(cfg)
    train_dataset = TrainDataset(
        img_id=train_df.image_id.values,
        image_dir=cfg.train_image_folder,
        target=train_df.classes.values,
        transform=get_train_transforms(cfg)
    )

    valid_dataset = TrainDataset(
        img_id=valid_df.image_id.values,
        image_dir=cfg.train_image_folder,
        target=valid_df.classes.values,
        transform=get_valid_transforms(cfg)
    )

    return train_dataset, valid_dataset


def make_data_loader(cfg):
    train_dataset, valid_dataset = build_dataset(cfg)

    if cfg.device == "TPU":
        batch_size = int(cfg.batch_size / 8)
    else:
        batch_size = cfg.batch_size

    if cfg.device == "TPU":
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            # pin_memory=False,
            sampler=train_sampler,
            drop_last=True,
            # num_workers=2,
            collate_fn=MixCollator(cfg)
        )
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            # pin_memory=False,
            drop_last=True,
            collate_fn=MixCollator(cfg)
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            # pin_memory=False,
            drop_last=True,
            # num_workers=2,
            collate_fn=MixCollator(cfg)
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            # pin_memory=False,
            drop_last=True,
            collate_fn=MixCollator(cfg)
        )

    return train_loader, valid_loader


def build_test_dataset(cfg):
    test_df = pd.read_csv(cfg.test_file)
    test_dataset = TestDataset(
        img_id=test_df.image_id.values,
        image_dir=cfg.test_image_folder,
        transform=get_test_transform(cfg)
    )
    return test_dataset


def make_test_data_loader(cfg):
    test_dataset = build_test_dataset(cfg)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        drop_last=False,
    )
    return test_loader










