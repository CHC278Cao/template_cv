# encoding: utf-8
"""
@author: ccj
@contact:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    print("TPU is not used.")


def mseloss_fn(outputs, targets):
    """
        MSELoss is applied to be as loss function
    :param outputs: Type: torch.tensor, shape is [bs, 1]
    :param targets: Type: torch.tensor, shape is [bs, 1]
    :return:
        the average mseloss
    """
    outputs = F.sigmoid(outputs)
    return nn.MSELoss()(outputs, targets.float())


def smooth_loss_fn(outputs, targets, beta, average_apply=True):
    """
        SMOOTHL1Loss is applied to be as loss function
    :param outputs: Type: torch.tensor, shape is [bs, 1]
    :param targets: Type: torch.tensor, shape is [bs, 1]
    :param beta: Type: torch.tensor
    :param average_apply: Type: bool, average the loss
    :return:
        the average smoothing l1 loss
    """
    outputs = F.sigmoid(outputs)
    n = torch.abs(outputs - targets.float())
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
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
    return nn.BCEWithLogitsLoss(outputs, targets.float())


def ce_loss_label_smoothing_fn(outputs, targets, num_classes, smoothing, device, label_weight=None):
    """
        Smoothing cross entropy is applied to be as loss function
    :param outputs: Type: torch.tensor, shape is [bs, *]
    :param targets: Type: torch.tensor, shape is [bs, *]
    :param num_classes: Type: int, number of classes
    :param smoothing: Type: float, smoothing coefficient for unbalanced labels
    :param device: Type: str, the type of device
    :param label_weight: Type: tuple, weights for labels
    :return:
        the average smooth cross entropy loss
    """
    if device == "TPU":
        device = xm.xla_device()
    elif device == "GPU":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    targets = targets.float()
    smooth_ohe = targets * (1 - smoothing) + (1 - targets) * smoothing / (num_classes - 1)
    outputs_log = F.log_softmax(outputs, dim=-1)
    loss = - outputs_log * smooth_ohe
    if label_weight is not None:
        if not isinstance(label_weight, torch.Tensor):
            label_weight = torch.tensor(label_weight, dtype=torch.float)
        label_weight = label_weight.to(device)
        loss = label_weight * loss
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
        # print("regression loss is applied in modeling")
        if cfg.criterion == "smooth-l1":
            return smooth_loss_fn(outputs, targets, cfg.l1_beta)
        elif cfg.criterion == "mse":
            return mseloss_fn(outputs, targets)
    elif output_size > 1 and cfg.ohe_mode:
        # print("classification loss is applied in modeling")
        if cfg.criterion == "cross-entropy":
            return ce_loss_fn(outputs, targets)
        elif cfg.criterion == "smooth-entropy":
            return ce_loss_label_smoothing_fn(outputs, targets, cfg.num_classes,
                                              cfg.smoothing, cfg.device, cfg.label_weight)


