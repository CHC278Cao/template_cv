# encoding: utf-8
"""
@author: ccj
@contact:
"""

import numpy as np

try:
    import torch
    import torch_xl.core.xla_model as xm
except ImportError:
    import torch


class EarlyStopping:
    def __init__(self, logger, device, patience=5, mode="max", delta=0.001):

        self.logger = logger
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.device = device
        self.delta = delta

        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.device == "TPU":
                xm.master_print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            else:
                self.logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            self.logger.info(f"Validation score improve ({self.val_score} --> {epoch_score}). Saving model!")

            if self.device == 'TPU':
                xm.master_print(f"Validation score improve ({self.val_score} --> {epoch_score}). Saving model!")
                xm.save(model.state_dict(), model_path)
            else:
                self.logger.info(f"Validation score improve ({self.val_score} --> {epoch_score}). Saving model!")
                torch.save(model.state_dict(), model_path)

            self.val_score = epoch_score

        else:
            if self.device == 'TPU':
                xm.master_print("epoch score is nan!")
            else:
                self.logger.info("epoch score is nan!")

    def get_stop(self):
        return self.early_stop