# encoding: utf-8
"""
@author: ccj
@contact:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mseloss_fn(outputs, targets):
    """
        MSELoss is applied to be as loss function
    :param outputs: Type: torch.tensor, shape is [bs, 1]
    :param targets: Type: torch.tensor, shape is [bs, 1]
    :return:
        the average mseloss
    """
    return nn.MSELoss()(outputs, targets.float())


def smooth_loss_fn(outputs, targets, cfg, average_apply=True):
    """
        SMOOTHL1Loss is applied to be as loss function
    :param outputs: Type: torch.tensor, shape is [bs, 1]
    :param targets: Type: torch.tensor, shape is [bs, 1]
    :param cfg: config file
    :param average_apply: Type: bool, average the loss
    :return:
        the average smoothing l1 loss
    """
    n = torch.abs(outputs - targets.float())
    cond = n < cfg.l1_beta
    loss = torch.where(cond, 0.5 * n ** 2 / cfg.l1_beta, n - 0.5 * cfg.beta)
    if average_apply:
        return loss.mean()
    else:
        return loss.sum()


def ce_loss_fn(outputs, targets):
    """
        Cross-entropy loss is applied to be as loss function
    :param outputs: Type: torch.tensor, shape is [bs, *]
    :param targets: Type: torch.tensor, shape is [bs, *]
    :return:
        the average cross entropy loss
    """
    return nn.BCEWithLogitsLoss(outputs, targets)


def ce_loss_label_smoothing_fn(outputs, targets, cfg):
    """
        Smoothing cross entropy is applied to be as loss function
    :param outputs: Type: torch.tensor, shape is [bs, *]
    :param targets: Type: torch.tensor, shape is [bs, *]
    :param cfg: config file
    :return:
        the average smooth cross entropy loss
    """
    smooth_ohe = targets * (1 - cfg.smoothing) + (1 - targets) * cfg.smoothing / (cfg.num_classes - 1)
    outputs_log = F.log_softmax(outputs, dim=-1)
    loss = - outputs_log * smooth_ohe
    if cfg.label_weight is not None:
        loss = cfg.label_weight * loss
    loss = torch.mean(loss)
    return loss


def loss_fn(outputs, targets, cfg):
    """
        Loss function for training
    :param outputs: Type: torch.tensor
    :param targets: Type: torch.tensor,
    :param cfg: config
    :return:
        the loss
    """
    assert (len(outputs.shape) == 2), "outputs should be 2 dimensions"

    output_size = outputs.shape[1]
    if output_size == 1 and not cfg.ohe_mode:
        print("regression loss is applied in modeling")
        if cfg.criterion == "smooth-l1":
            return smooth_loss_fn(outputs, targets, cfg)
        elif cfg.criterion == "mse":
            return mseloss_fn(outputs, targets)
    elif output_size > 1 and cfg.ohe_mode:
        print("classification loss is applied in modeling")
        if cfg.criterion == "cross-entropy":
            return ce_loss_fn(outputs, targets)
        elif cfg.criterion == "smooth-entropy":
            return ce_loss_label_smoothing_fn(outputs, targets, cfg)


