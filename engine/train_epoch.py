# encoding: utf-8
"""
@author: ccj
@contact:
"""

import gc

import torch.nn.functional as F
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    import torch

from utils.meter import AverageMeter
from .loss import loss_fn
from .utils import reduce_fn


def train_epoch(model, data_loader, optimizer, cfg, scheduler=None):
    """
        The training process for one epoch
    :param model: Type:
    :param data_loader: Type: DataLoader, the training dataloader
    :param optimizer: Type: Optim.optimizer, the learning optimizer
    :param cfg: Type: config, the config file
    :param scheduler: Type: lr_scheduler, the learning scheduler
    :return:
        the average loss for training
    """
    model.train()

    if cfg.device == "TPU":
        device = xm.xla_device()
    elif cfg.device == "GPU":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    loss_meter = AverageMeter()
    output_fin = []
    target_fin = []

    for batch_idx, data in enumerate(data_loader):
        image = data["image"].to(device)
        target = data["target"].to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, target, cfg=cfg)
        loss.backward()
        if cfg.device == "TPU":
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if cfg.device == "TPU":
            reduced_loss = xm.mesh_reduce("reduce_loss", loss, reduce_fn)
            loss_meter.update(reduced_loss.item(), image.shape[0])
        else:
            loss_meter.update(loss.item(), image.shape[0])

        if 1 == output.shape[-1]:
            output_fin.extend(F.sigmoid(output).detach().squeeze().cpu().numpy().tolist())
            target_fin.extend(target.detach().cpu().numpy().tolist())
        else:
            # output_fin.extend(torch.argmax(output, dim=-1).detach().squeeze().cpu().numpy().tolist())
            output_fin.extend(F.softmax(output, dim=-1).detach().squeeze().cpu().numpy()[:, 1].tolist())
            if cfg.ohe_mode:
                target_fin.extend(torch.argmax(target, dim=-1).detach().squeeze().cpu().numpy().tolist())
            else:
                target_fin.extend(target.detach.cpu().numpy().tolist())

        if batch_idx % 100 == 99:
            if cfg.device == "TPU":
                xm.master_print(f"In training, Batch Idx = {batch_idx + 1}, Loss = {loss_meter.get_avg()}")
            else:
                print(f"In training, Batch Idx = {batch_idx + 1}, Loss = {loss_meter.get_avg()}")

        del data
        gc.collect()
        torch.cuda.empty_cache()

    return loss_meter.get_avg(), output_fin, target_fin









