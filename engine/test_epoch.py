# encoding: utf-8
"""
@author: ccj
@contact:
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def test_epoch(model, data_loader, cfg):
    model.eval()
    output_fin = []

    if cfg.device == "TPU":
        raise Exception("TPU is not available fot prediction.")
    elif cfg.device == "GPU":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for data in tk0:
            image = data["image"].to(device)
            output = model(image)
            if 1 == output.shape[-1]:
                output_fin.extend(F.sigmoid(output).detach().squeeze().cpu().numpy().tolist())
            else:
                # output_fin.extend(torch.argmax(output, dim=-1).detach().squeeze().cpu().numpy().tolist())
                output_fin.extend(F.softmax(output, dim=-1).detach().squeeze().cpu().numpy()[:, 1].tolist())

    return output_fin


