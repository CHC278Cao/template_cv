# encoding: utf-8
"""
@author: ccj
@contact:
"""


try:
    import torch_xla.core.model_xla as xm
    import torch_xla.distributed.parallel_loader as pl
except ImportError:
    import torch


from solver.optimizer import make_optimizer
from solver.scheduler import make_scheduler
from .train_epoch import train_epoch
from .valid_epoch import valid_epoch
from .test_epoch import predict



class Fitter:
    def __init__(self, model, optimizer, cfg, logger, averager, schduler=None):
        """

        :param model:
        :param optimizer:
        :param cfg:
        :param logger:
        :param schduler:
        """
        super(Fitter, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.logger = logger
        self.averager = averager
        self.scheduler = schduler

        if cfg.device == "TPU":
            self.device = xm.xla_device()
        elif cfg.device == "GPU":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        self.lr = []

    def train(self, train_data_loader, valid_data_loader):

        optimizer = make_optimizer(cfg=self.cfg, model=self.model)
        scheduler = make_scheduler(cfg=self.cfg, optimizer=optimizer)

        for epoch in range(self.cfg.epochs):
            if self.cfg.device == "TPU":
                train_para_loader = pl.ParallelLoader(train_data_loader, [self.device])
                train_data_loader = train_para_loader.per_device_loader(self.device)
                valid_para_loader = pl.ParallelLoader(valid_data_loader, [self.device])
                valid_data_loader = valid_para_loader.per_device_loader(self.device)

            self.lr.append(optimizer.param_group[0]['lr'])

            train_loss, train_output, train_target = train_epoch(self.model, train_data_loader, optimizer, self.cfg)
            valid_loss, valid_output, valid_target = valid_epoch(self.model, valid_data_loader, self.cfg)
            scheduler.step()


















