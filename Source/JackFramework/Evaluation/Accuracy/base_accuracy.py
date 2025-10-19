# -*- coding: utf-8 -*-
import torch

try:
    from ._meta_accuracy import MetaAccuracy
except ImportError:
    from _meta_accuracy import MetaAccuracy


class BaseAccuracy(MetaAccuracy):
    """Common accuracy utilities used across tasks."""

    @staticmethod
    def rmse_score(res: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if res.shape != gt.shape:
            raise ValueError('Prediction and ground truth tensors must share the same shape.')
        res_flat = res.reshape(res.size(0), -1)
        gt_flat = gt.reshape(gt.size(0), -1)
        rmse = torch.sqrt(torch.sum((res_flat - gt_flat) ** 2, dim=1) / res_flat.size(1))
        return torch.mean(rmse)
