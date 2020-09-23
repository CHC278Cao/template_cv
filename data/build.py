# encoding: utf-8
"""
@author: ccj
@contact:
    When training, adjust the split_dataset function to get available data
"""

import os
import pdb
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
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


def split_dataset(hparams: Dict[str, Any]) -> pd.DataFrame:
    """
        Add split index to label the train and valid dataset
    :param hparams:
    :return:
    """
    df = pd.read_csv(hparams["data_file"])
    df = df.sample(frac=1., random_state=hparams["seed"]).reset_index(drop=True)

    df["classes"] = df["healthy"] * 0 + df["multiple_diseases"] * 1 \
                    + df["rust"] * 2 + df["scab"] * 3
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=hparams["dataset_parameters"]["n_folds"], shuffle=True,
                          random_state=hparams["seed"])
    for idx, (train_idx, valid_idx) in enumerate(skf.split(X=df, y=df["classes"].values)):
        df.loc[valid_idx, "fold"] = idx

    return df


def build_dataset(hparams: Dict[str, Any]) -> Tuple[]:
    df = split_dataset(hparams)

    train_df = df[df["fold"] != hparams["fold_idx"]].reset_index(drop=True)
    valid_df = df[df["fold"] == hparams["fold_idx"]].reset_index(drop=True)

    train_df.to_csv(f"{hparams['train_file']}", index=False)
    valid_df.to_csv(f"{hparams['valid_file']}", index=False)

    train_dataset = TrainDataset(
        img_id=train_df.image_id.values,
        image_dir=hparams["train_image_folder"],
        target=train_df.classes.values,
        transform=get_train_transforms(hparams)
    )

    valid_dataset = TrainDataset(
        img_id=valid_df.image_id.values,
        image_dir=hparams["train_image_folder"],
        target=valid_df.classes.values,
        transform=get_valid_transforms(hparams)
    )

    return train_dataset, valid_dataset


def make_data_loader(hparams):
    train_dataset, valid_dataset = build_dataset(hparams)

    if hparams["device"] == "TPU":
        batch_size = int(hparams.batch_size / 8)
    else:
        batch_size = hparams.batch_size

    if hparams["device"] == "TPU":
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
            collate_fn=MixCollator(hparams)
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
            collate_fn=MixCollator(hparams)
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            # pin_memory=False,
            drop_last=True,
            # num_workers=2,
            collate_fn=MixCollator(hparams)
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            # pin_memory=False,
            drop_last=True,
            collate_fn=MixCollator(hparams)
        )

    return train_loader, valid_loader


def build_test_dataset(hparams):
    test_df = pd.read_csv(hparams["test_file"])
    test_dataset = TestDataset(
        img_id=test_df.image_id.values,
        image_dir=hparams["test_image_folder"],
        transform=get_test_transform(hparams)
    )
    return test_dataset


def make_test_data_loader(hparams):
    test_dataset = build_test_dataset(hparams)

    test_loader = DataLoader(
        test_dataset,
        batch_size=hparams.test_batch_size,
        shuffle=False,
        drop_last=False,
    )
    return test_loader










