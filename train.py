# encoding: utf-8
"""
@author: ccj
@contact:
"""

import os
# import apex
import pdb
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from albumentations.core.serialization import from_dict

from pathlib import Path
from typing import List, Dict, Any
from collections import OrderedDict

try:
    import torch_xla.core.xla_model as xm
    TPU_IS_AVALIABLE = True
except ImportError:
    TPU_IS_AVALIABLE = False

import torch
# from data import build
# from .engine.utils import load_checkpoint
# from .utils.logger import init_logger
# from .utils.early_stopping import EarlyStopping
# from .utils.read_dict import object_from_dict
from .utils.set_seed import seed_everything



# def add_device_arg(parser, name, helpstring='device to run modeling'):
#     try:
#         import torch_xla.core.xla_model as xm
#         device = "TPU"
#     except ImportError:
#         device = 'GPU' if torch.cuda.is_available() else 'CPU'
#
#     dest_name = name.replace('-', '_')
#     parser.add_argument('--' + name, dest=dest_name, default=device, help=helpstring)


def get_args():
    parser = argparse.ArgumentParser(description="Image Model Training")
    parser.add_argument('--config-path', type=Path, help="Path to the config", required=True)
    return parser.parse_args()


# Dataset / Model parameters
# parser.add_argument('--train-image-folder', metavar='DIR',
#                     help='path to training image folder')
# parser.add_argument('--train-file', metavar='FILE', help='path to training file')
# parser.add_argument('--test-image-folder', metavar='DIR',
#                     help='path to testing image folder')
# parser.add_argument('--test-file', metavar='FILE', help='path to testing file')
# parser.add_argument('--n-folds', type=int, default=5, metavar='FOLDS',
#                     help='number of folds to split data (default: 5)')
# parser.add_argument('--fold-idx', type=int, metavar='FOLD',
#                     help='fold index to valid')
# parser.add_argument('--pretrained', action='store_true', default=False,
#                     help='start with pretrained modeling(if avail)')
# parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
#                     help='Initialize modeling from this checkpoint(default: none)')
# parser.add_argument('--num-class', type=int, metavar='N',
#                     help='total number of classes for classification')
# parser.add_argument('--ohe-mode', action='store_true', default=False,
#                     help='apply one hot vector for target')
# parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
#                     help='input batch size for training (default: 8)')
# parser.add_argument('-tb', '--test-batch-size', type=int, default=2, metavar='N',
#                     help='input batch size for test (default: 2)')
# parser.add_argument('--dropout', type=float, default=0.15, metavar='PCT',
#                     help='Dropout rate (default: 0.15)')
# parser.add_argument('--clip-grad', type=float, default=6.0, metavar='NORM',
#                     help='Clip gradient norm (default: 6.0)')
# add_device_arg(parser, name='device')
# parser.add_argument('--criterion', type=str, metavar='CRITERION',
#                     help='criterion for loss function ("smooth-l1", "mse", "cross-entropy", "smooth-entropy"')
# parser.add_argument('--l1-beta', type=float, default=1.5,
#                     help='smoothing beta for l1 loss (default: 1.5)')
# parser.add_argument('--smoothing', type=float, default=0.1, metavar='SMOOTH',
#                     help='smoothing efficiency to label')
# parser.add_argument('--label-weight', type=tuple,
#                     help='label weights for smoothing cross entropy')
#

#
# # Learning rate scheduler
# parser.add_argument('--sched', type=str, default='LambdaLR', metavar='SCHEDULER',
#                     help='LR scheduler (default: lr_lambda)')
# parser.add_argument('--lr-start', type=float, default=1e-4, metavar='LR_START',
#                     help='lr start for LR scheduler (default: 1e-4)')
# parser.add_argument('--lr-end', type=float, default=2e-3, metavar='LR_END',
#                     help='lr end for LR scheduler (default: 2e-3)')
# parser.add_argument('--warm-up-ratio', type=float, default=0.4, metavar='WARMUP',
#                     help='warm up ratio to total epochs (default: 0.4)')
# parser.add_argument('--sustain-ratio', type=float, default=0., metavar='SUSTAIN',
#                     help='sustain ratio to total epochs (default: 0.)')
# parser.add_argument('--exp', type=float, default=0.8, metavar="EXP",
#                     help='exponential rate for decay (default: 0.8)')
# parser.add_argument('--epochs', type=int, default=60, metavar='EPOCHS',
#                     help='number of epochs to train (default: 60)')
#


#
# # Misc
# parser.add_argument('--project-name', type=str, metavar="PROJECT",
#                     help='name for this project')
# parser.add_argument('--log-file', type=str, default='./outputs/train.log', metavar='PATH',
#                     help='logger file path (default: train.log)')
# parser.add_argument('--output-dir', type=str, default='./outputs',
#                     help='directory to store output')
# parser.add_argument('--seed', type=int, default=42, metavar='S',
#                     help='random seed (default: 42)')
# parser.add_argument('--eval-metric', )
# parser.add_argument('--eval-mode', type=str, default='min',
#                     help='set eval mode to test valid data (default: "min")')
# parser.add_argument('--eval-patience', type=int, default=5,
#                     help='number of patience to ignore bad score')
# parser.add_argument('--eval-delta', type=float, default=0.001,
#                     help='the minimum score to be achieved in two contiguous epochs (default: 0.001)')

#
# class Fitter:
#     def __init__(self, model, hparams, logger, earlystopping):
#         """
#
#         :param model:
#         :param hparams:
#         :param logger:
#         :param meter_averager:
#         """
#         super(Fitter, self).__init__()
#         self.model = model
#         self.hparams = hparams
#         self.logger = logger
#         self.earlystopping = earlystopping
#         self.optimizer = None
#
#         corrections: Dict[str, str] = {"model.": ""}
#
#         if "weights" in self.hparams:
#             checkpoint = load_checkpoint(file_path=self.hparams["weights"], rename_in_layers=corrections)
#             self.model.load_state_dict(checkpoint["state_dict"])
#
#         if self.hparams["sync_bn"]:
#             self.model = apex.parallel.convert_syncbn_model(self.model)
#
#         if self.hparams["device"] == "TPU":
#             self.device = xm.xla_device()
#         elif self.hparams["device"] == "GPU":
#             self.device = torch.device("cuda")
#         else:
#             self.device = torch.device("cpu")
#
#         self.model.to(self.device)
#
#         self.lr = []
#
#     def configure_optimizer(self):
#         optimizer = object_from_dict(self.hparams["optimizer"],
#                                      params=[x for x in self.model.parameters() if x.requires_grad])
#         scheduler = object_from_dict(self.hparams["scheduler"],
#                                      optimizer=optimizer)
#         self.optimizer = [optimizer]
#         return self.optimizer, [scheduler]
#
#     def train_
#
#     def train(self, train_data_loader, valid_data_loader, optimizer, scheduler):
#
#         for epoch in range(self.hparams.epochs):
#             if self.hparams["device"] == "TPU":
#                 train_para_loader = pl.ParallelLoader(train_data_loader, [self.device])
#                 train_data_loader = train_para_loader.per_device_loader(self.device)
#                 valid_para_loader = pl.ParallelLoader(valid_data_loader, [self.device])
#                 valid_data_loader = valid_para_loader.per_device_loader(self.device)
#
#             self.lr.append(optimizer.param_group[0]['lr'])
#
#             train_loss, train_outputs, train_targets = train_epoch(self.model, train_data_loader, optimizer, self.hparams)
#             valid_loss, valid_outputs, valid_targets = valid_epoch(self.model, valid_data_loader, self.hparams)
#
#             if optimizer.__class__.__name__ == "ReduceLROnPlateau":
#                 scheduler.step(valid_loss)
#             else:
#                 scheduler.step()
#
#             train_accuracy = metrics.accuracy_score(np.round(train_targets).astype(np.int),
#                                                     np.round(train_outputs).astype(np.int))
#             train_auc = metrics.roc_auc_score(np.round(train_targets).astype(np.int), train_outputs)
#             valid_accuracy = metrics.accuracy_score(np.round(valid_targets).astype(np.int),
#                                                     np.round(valid_outputs).astype(np.int))
#             valid_auc = metrics.roc_auc_score(np.round(valid_targets).astype(np.int), valid_outputs)
#
#             if self.hparams.device == "TPU":
#                 self.logger.info(f"device = {xm.get_ordinal()}, train_acc = {train_accuracy}, train_auc = {train_auc}")
#                 self.logger.info(f"device = {xm.get_ordinal()}, valid_acc = {valid_accuracy}, valid_auc = {valid_auc}")
#                 train_fin_acc = xm.mesh_reduce("train_reduce_acc", train_accuracy, reduce_fn=reduce_fn)
#                 train_fin_auc = xm.mesh_reduce("train_reduce_auc", train_auc, reduce_fn=reduce_fn)
#                 valid_fin_acc = xm.mesh_reduce("valid_reduce_acc", valid_accuracy, reduce_fn=reduce_fn)
#                 valid_fin_auc = xm.mesh_reduce("valid_reduce_auc", valid_auc, reduce_fn=reduce_fn)
#                 xm.master_print(f"Epoch = {epoch + 1}, train_acc = {train_fin_acc}, train_auc = {train_fin_auc}")
#                 xm.master_print(f"Epoch = {epoch + 1}, valid_acc = {valid_fin_acc}, valid_auc = {valid_fin_auc}")
#
#             else:
#                 train_fin_acc = train_accuracy
#                 train_fin_auc = train_auc
#                 valid_fin_acc = valid_accuracy
#                 valid_fin_auc = valid_auc
#                 print(f"Epoch = {epoch + 1}, train_acc = {train_fin_acc}, train_auc = {train_fin_auc}")
#                 print(f"Epoch = {epoch + 1}, valid_acc = {valid_fin_acc}, valid_auc = {valid_fin_auc}")
#
#             self.earlystopping(valid_fin_auc, self.model, f"model_{self.hparams.fold_idx}.bin")
#
#             if self.earlystopping.get_stop():
#                 break
#
#
#     def predict(self, model, data_loader):
#         pass

def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    seed_everything(hparams["seed"])

    # print(hparams)
    transform = from_dict(hparams["train_aug"])
    print(transform)




    # train_loader, valid_loader = build.make_data_loader(cfg)
    # # model =
    #
    # logger = init_logger(cfg.project_name, cfg.log_file)
    # early_stopping = EarlyStopping(logger, cfg.device, cfg.eval_patience, cfg.eval_mode, cfg.eval_delta)
    #
    #
    # # fitter =

if __name__ == "__main__":
    main()