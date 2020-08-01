# encoding: utf-8
"""
@author: ccj
@contact:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: f(X) = pow(sum(pow(X, p)), 1/p)
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    reference: see details at https://arxiv.org/pdf/1711.02512.pdf
    Args:
        p: Type float, power coefficient
        eps: Type float, low bound to avoid extreme small number
    """
    def __init__(self, p=3., eps=1e-6):
        super(GeM, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        x = torch.clamp(x, min=self.eps).pow(self.p)
        out = F.adaptive_avg_pool2d(x, output_size=1).pow(1. / self.p)
        return out

    def __repr__(self):
        return self.__class__.__name__ \
               + '(' + 'p=' + str(self.p) + ')' \
               + '(' + 'eps' + str(self.eps) + ')'

