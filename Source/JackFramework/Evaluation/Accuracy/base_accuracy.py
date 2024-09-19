# -*- coding: utf-8 -*-
import torch
try:
    from ._meta_accuracy import MetaAccuracy
except ImportError:
    from _meta_accuracy import MetaAccuracy


class BaseAccuracy(MetaAccuracy):
    """docstring for common accuracy"""
    __BASE_ACCURACY_INSTANCE = None

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__BASE_ACCURACY_INSTANCE is None:
            cls.__BASE_ACCURACY_INSTANCE = object.__new__(cls)
        return cls.__BASE_ACCURACY_INSTANCE

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def rmse_score(res: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        res = res.view(res.size(0), -1)
        gt = gt.view(gt.size(0), -1)
        rmse = torch.sqrt(torch.sum((res - gt) ** 2, dim=1) / res.size(1))
        return torch.mean(rmse)
