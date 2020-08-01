# encoding: utf-8
"""
@author: ccj
@contact:
"""

import torch
from tqdm import tqdm


def predict(model, data_loader, cfg):
    model.eval()
    output_fin = []

    if cfg.device == "TPU":
        raise Exception("TPU is not available fot prediction.")
    elif cfg.device == "GPU":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with torch.no_grad():
        tk0 = tqdm(data_loader, len(data_loader))
        for data in tk0:
            image = data["image"].to(device)
            output = model(image)
            if 1 == output.shape[-1]:
                output_fin.append(output.detach().cpu().numpy().tolist())
            else:
                output_fin.append(torch.argmax(output).detach().cpu().numpy().tolist())

    return output_fin


