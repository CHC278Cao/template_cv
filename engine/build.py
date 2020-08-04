# encoding: utf-8
"""
@author: ccj
@contact:
"""

import numpy as np

from sklearn import metrics

try:
    import torch_xla.core.model_xla as xm
    import torch_xla.distributed.parallel_loader as pl
except ImportError:
    import torch


from solver.optimizer import make_optimizer
from solver.scheduler import make_scheduler
from engine.utils import reduce_fn
from .train_epoch import train_epoch
from .valid_epoch import valid_epoch
from .test_epoch import predict


class Fitter:
    def __init__(self, model, cfg, logger, earlystopping):
        """

        :param model:
        :param cfg:
        :param logger:
        :param meter_averager:
        """
        super(Fitter, self).__init__()
        self.model = model
        self.cfg = cfg
        self.logger = logger
        self.earlystopping = earlystopping

        if cfg.device == "TPU":
            self.device = xm.xla_device()
        elif cfg.device == "GPU":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        self.lr = []

    def train(self, train_data_loader, valid_data_loader, optimizer, scheduler):

        for epoch in range(self.cfg.epochs):
            if self.cfg.device == "TPU":
                train_para_loader = pl.ParallelLoader(train_data_loader, [self.device])
                train_data_loader = train_para_loader.per_device_loader(self.device)
                valid_para_loader = pl.ParallelLoader(valid_data_loader, [self.device])
                valid_data_loader = valid_para_loader.per_device_loader(self.device)

            self.lr.append(optimizer.param_group[0]['lr'])

            train_loss, train_outputs, train_targets = train_epoch(self.model, train_data_loader, optimizer, self.cfg)
            valid_loss, valid_outputs, valid_targets = valid_epoch(self.model, valid_data_loader, self.cfg)

            if optimizer.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(valid_loss)
            else:
                scheduler.step()

            train_accuracy = metrics.accuracy_score(np.round(train_targets).astype(np.int),
                                                    np.round(train_outputs).astype(np.int))
            train_auc = metrics.roc_auc_score(np.round(train_targets).astype(np.int), train_outputs)
            valid_accuracy = metrics.accuracy_score(np.round(valid_targets).astype(np.int),
                                                    np.round(valid_outputs).astype(np.int))
            valid_auc = metrics.roc_auc_score(np.round(valid_targets).astype(np.int), valid_outputs)

            if self.cfg.device == "TPU":
                self.logger.info(f"device = {xm.get_ordinal()}, train_acc = {train_accuracy}, train_auc = {train_auc}")
                self.logger.info(f"device = {xm.get_ordinal()}, valid_acc = {valid_accuracy}, valid_auc = {valid_auc}")
                train_fin_acc = xm.mesh_reduce("train_reduce_acc", train_accuracy, reduce_fn=reduce_fn)
                train_fin_auc = xm.mesh_reduce("train_reduce_auc", train_auc, reduce_fn=reduce_fn)
                valid_fin_acc = xm.mesh_reduce("valid_reduce_acc", valid_accuracy, reduce_fn=reduce_fn)
                valid_fin_auc = xm.mesh_reduce("valid_reduce_auc", valid_auc, reduce_fn=reduce_fn)
                xm.master_print(f"Epoch = {epoch + 1}, train_acc = {train_fin_acc}, train_auc = {train_fin_auc}")
                xm.master_print(f"Epoch = {epoch + 1}, valid_acc = {valid_fin_acc}, valid_auc = {valid_fin_auc}")

            else:
                train_fin_acc = train_accuracy
                train_fin_auc = train_auc
                valid_fin_acc = valid_accuracy
                valid_fin_auc = valid_auc
                print(f"Epoch = {epoch + 1}, train_acc = {train_fin_acc}, train_auc = {train_fin_auc}")
                print(f"Epoch = {epoch + 1}, valid_acc = {valid_fin_acc}, valid_auc = {valid_fin_auc}")

            self.earlystopping(valid_fin_auc, self.model, f"model_{self.cfg.fold_idx}.bin")




















