# encoding: utf-8
"""
@author: ccj
@contact:
"""

import os
import numpy as np
import pandas as pd
from sklearn import model_selection


def create_train_data(cfg):
    df = pd.read_csv(cfg.train_file)

