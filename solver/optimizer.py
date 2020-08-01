# encoding: utf-8
"""
@author: ccj
@contact:
"""

import torch


def make_optimizer(cfg, model):
    optimizer = getattr(torch.optim, cfg.opt)(model.parameters(), lr=cfg.lr,
                                              eps=cfg.opt_eps, weight_decay=cfg.weight_decay)
    return optimizer
