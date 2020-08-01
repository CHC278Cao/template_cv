# encoding: utf-8
"""
@author: ccj
@contact:
"""

import os
import pdb
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from data import build
from .utils.logger import init_logger
from .utils.early_stopping import EarlyStopping
from .engine.build import Fitter


def add_device_arg(parser, name, help='device to run modeling'):
    try:
        import torch_xla.core.xla_model as xm
        device = "TPU"
    except ImportError:
        device = 'GPU' if torch.cuda.is_available() else 'CPU'

    dest_name = name.replace('-', '_')
    parser.add_argument('--' + name, dest=dest_name, default=device, help=help)


parser = argparse.ArgumentParser(description="Image Model Training")
# Dataset / Model parameters
parser.add_argument('--train-image-folder', metavar='DIR',
                    help='path to training image folder')
parser.add_argument('--train-file', metavar='FILE', help='path to training file')
parser.add_argument('--test-image-folder', metavar='DIR',
                    help='path to testing image folder')
parser.add_argument('--test-file', metavar='FILE', help='path to testing file')
parser.add_argument('--n-folds', type=int, default=5, metavar='FOLDS',
                    help='number of folds to split data (default: 5)')
parser.add_argument('--fold-idx', type=int, metavar='FOLD',
                    help='fold index to valid')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='start with pretrained modeling(if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize modeling from this checkpoint(default: none)')
parser.add_argument('--num-class', type=int, metavar='N',
                    help='total number of classes for classification')
parser.add_argument('--ohe-mode', action='store_true', default=False,
                    help='apply one hot vector for target')
parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('-tb', '--test-batch-size', type=int, default=2, metavar='N',
                    help='input batch size for test (default: 2)')
parser.add_argument('--dropout', type=float, default=0.15, metavar='PCT',
                    help='Dropout rate (default: 0.15)')
parser.add_argument('--clip-grad', type=float, default=6.0, metavar='NORM',
                    help='Clip gradient norm (default: 6.0)')
add_device_arg(parser, name='device')
parser.add_argument('--criterion', type=str, metavar='CRITERION',
                    help='criterion for loss function ("smooth-l1", "mse", "cross-entropy", "smooth-entropy"')
parser.add_argument('--l1-beta', type=float, default=1.5,
                    help='smoothing beta for l1 loss (default: 1.5)')
parser.add_argument('--smoothing', type=float, default=0.1, metavar='SMOOTH',
                    help='smoothing efficiency to label')
parser.add_argument('--label-weight', type=list,
                    help='label weights for smoothing cross entropy')

# Optimizer parameters
parser.add_argument('--opt', type=str, default='Adam', metavar='OPTIMIZER',
                    help='Optimizer (default: Adam)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--opt-eps', type=float, default=1e-3, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1-5)')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='DECAY',
                    help='weight decay (default: 1e-4)')

# Learning rate scheduler
parser.add_argument('--sched', type=str, default='LambdaLR', metavar='SCHEDULER',
                    help='LR scheduler (default: lr_lambda)')
parser.add_argument('--lr-start', type=float, default=1e-4, metavar='LR_START',
                    help='lr start for LR scheduler (default: 1e-4)')
parser.add_argument('--lr-end', type=float, default=2e-3, metavar='LR_END',
                    help='lr end for LR scheduler (default: 2e-3)')
parser.add_argument('--warm-up-ratio', type=float, default=0.4, metavar='WARMUP',
                    help='warm up ratio to total epochs (default: 0.4)')
parser.add_argument('--sustain-ratio', type=float, default=0., metavar='SUSTAIN',
                    help='sustain ratio to total epochs (default: 0.)')
parser.add_argument('--exp', type=float, default=0.8, metavar="EXP",
                    help='exponential rate for decay (default: 0.8)')
parser.add_argument('--epochs', type=int, default=60, metavar='EPOCHS',
                    help='number of epochs to train (default: 60)')

# Augmentation parameters
parser.add_argument('--mean', type=tuple, default=(0.485, 0.456, 0.406), metavar="MEAN",
                    help='mean pixel value of data')
parser.add_argument('--std', type=tuple, default=(0.229, 0.224, 0.225), metavar='STD',
                    help='std pixel value of data')
parser.add_argument('--hflip-prob', type=float, default=0.5, metavar='HFLIP',
                    help="probability to horizontally flip images (default: 0.5)")
parser.add_argument('--vflip-prob', type=float, default=0.5, metavar='VFLIP',
                    help='probability to vertically flip image (default: 0.5)')
parser.add_argument('--hue-limit', type=int, default=20,
                    help='hue shift limit for HueSaturationValue (default: 20)')
parser.add_argument('--sat-limit', type=int, default=30,
                    help='sat shift limit for HueSaturationValue (default: 30)')
parser.add_argument('--val-limit', type=int, default=20,
                    help='val shift limit for HueSaturationValue (default: 20)')
parser.add_argument('--hue-prob', type=float, default=0.4,
                    help='the probability for HueSaturationValue (default: 0.4)')
parser.add_argument('--brightness-limit', type=float, default=0.3,
                    help='brightness limit for RandomBrightnessContrast (default: 0.3)')
parser.add_argument('--contrast-limit', type=float, default=0.2,
                    help='contrast limit for RandomBrightnessContrast (default: 0.2)')
parser.add_argument('--contrast-prob', type=float, default=0.4,
                    help='the probability for RandomBrightnessContrast (default: 0.4)')
parser.add_argument('--cutmix-mode', action='store_false', default=True,
                    help="apply cutmix transform to dataset (default: True)")
parser.add_argument('--cutmix-prob', type=float, default=0.5,
                    help='the probability of applying cutmix transform')
parser.add_argument('--mixup-mode', action='store_true', default=False,
                    help="apply mixup transform to dataset (default: True)")
parser.add_argument('--mixup-prob', type=float, default=0.5,
                    help='the probability of applying mixup transform')
parser.add_argument('--gray_prob', type=float, default=0.4,
                    help='the probability for togray (default: 0.4)')
parser.add_argument('--coarse-max-holes', type=int, default=8,
                    help='max number of holes for coarse dropout (default: 8)')
parser.add_argument('--coarse-max-height', type=int, default=16,
                    help='max height of height for coarse dropout (default: 16)')
parser.add_argument('--coarse-max-width', type=int, default=16,
                    help='max height of width for coarse dropout (default: 16)')
parser.add_argument('--coarse-prob', type=float, default=0.3,
                    help='the probability for coarse dropout (default: 0.3)')

# Misc
parser.add_argument('--project-name', type=str, metavar="PROJECT",
                    help='name for this project')
parser.add_argument('--log-file', type=str, default='./outputs/train.log', metavar='PATH',
                    help='logger file path (default: train.log)')
parser.add_argument('--output-dir', type=str, default='./outputs',
                    help='directory to store output')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--eval-metric', )
parser.add_argument('--eval-mode', type=str, default='min',
                    help='set eval mode to test valid data (default: "min")')
parser.add_argument('--eval-patience', type=int, default=5,
                    help='number of patience to ignore bad score')
parser.add_argument('--eval-delta', type=float, default=0.001,
                    help='the minimum score to be achieved in two contiguous epochs (default: 0.001)')



def main():
    cfg = parser.parse_args()
    train_loader, valid_loader = build.make_data_loader(cfg)
    # model =

    logger = init_logger(cfg)
    early_stopping = EarlyStopping(logger, cfg)


    fitter =

if __name__ == "__main__":
    main()