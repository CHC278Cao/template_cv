# encoding: utf-8
"""
@author: ccj
@contact:
"""

import re
from pathlib import Path
from typing import Union, Optional, Any, Dict, List

import torch


def reduce_fn(vals):
    """
        Reduce function to average values when using TPU
    :param vals: value
    :return:
        the average value
    """
    return sum(vals) / len(vals)


def load_checkpoint(file_path: Union[Path, str], rename_in_layers: Optional[dict] = None) -> Dict[str, Any]:
    """
        Load pretrained weights from file
    :param file_path: pretrained weight file path
    :param rename_in_layers: layer rename
    :return:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if rename_in_layers is not None:
        model_state_dict = checkpoint["state_dict"]

        result = {}
        for key, value in model_state_dict.items():
            for key_r, value_r in rename_in_layers.items():
                key = re.sub(key_r, value_r, key)
            result[key] = value

    return checkpoint


