# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F  # (uncomment if needed,but you likely already have it)


# Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
# https://arxiv.org/abs/1908.08681v1
# implemented for PyTorch / FastAI by lessw2020
# github: https://github.com/lessw2020/mish


class Mish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        # inlining this saves 1 second per epoch (V100 GPU) vs
        # having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))
