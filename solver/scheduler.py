# encoding: utf-8
"""
@author: ccj
@contact:
"""

import torch


def poly_lr_fn(epoch, cfg):
    warmup = int(cfg.warm_up_ratio * cfg.epochs)
    sustain = int(cfg.sustain_ratio * cfg.epochs)
    if epoch < warmup:
        lr = (cfg.lr_end - cfg.lr_start) / warmup * epoch
    elif epoch < warmup + sustain:
        lr = cfg.lr_end
    else:
        lr = (cfg.lr_end - cfg.lr_start) * cfg.exp ** (epoch - warmup - sustain) + cfg.lr_start

    return lr


def make_scheduler(cfg, optimizer):
    scheduler = getattr(torch.optim.lr_scheduler, cfg.sched)(optimizer, lr_lambda=poly_lr_fn)
    return scheduler
