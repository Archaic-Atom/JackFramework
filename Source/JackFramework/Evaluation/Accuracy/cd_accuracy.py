# -*- coding: utf-8 -*-
import torch
from typing import Optional, Tuple

try:
    from ._meta_accuracy import MetaAccuracy
except ImportError:
    from _meta_accuracy import MetaAccuracy


class CDAccuracy(MetaAccuracy):
    """Change detection metrics using cached confusion matrices."""

    _CONFUSION_MATRIX: Optional[torch.Tensor] = None
    _LAST_CONFUSION_MATRIX: Optional[torch.Tensor] = None

    @classmethod
    def reset_cm(cls) -> None:
        cls._CONFUSION_MATRIX = None
        cls._LAST_CONFUSION_MATRIX = None

    @classmethod
    def generate_confusion_matrix(cls, res: torch.Tensor, gt: torch.Tensor,
                                  num_classes: int, accumulate: bool = True) -> torch.Tensor:
        res = res.flatten().to(torch.int64)
        gt = gt.flatten().to(torch.int64)
        mask = (gt >= 0) & (gt < num_classes)
        cm = torch.bincount(
            num_classes * gt[mask] + res[mask],
            minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes)
        cls._LAST_CONFUSION_MATRIX = cm
        if accumulate:
            cls._CONFUSION_MATRIX = cm if cls._CONFUSION_MATRIX is None else cls._CONFUSION_MATRIX + cm
        return cm

    @classmethod
    def _select_cm(cls, accumulate: bool) -> torch.Tensor:
        cm = cls._CONFUSION_MATRIX if accumulate else cls._LAST_CONFUSION_MATRIX
        if cm is None:
            raise RuntimeError('Confusion matrix not generated yet. Call `generate_confusion_matrix` first.')
        return cm.to(torch.float32)

    @classmethod
    def oa_score(cls, accumulate: bool = False) -> torch.Tensor:
        cm = cls._select_cm(accumulate)
        tp = torch.diag(cm)
        return tp.sum() / (cm.sum() + cls.epsilon)

    @classmethod
    def precision_score(cls, accumulate: bool = False) -> torch.Tensor:
        cm = cls._select_cm(accumulate)
        tp = torch.diag(cm)
        return tp / (cm.sum(dim=0) + cls.epsilon)

    @classmethod
    def recall_score(cls, accumulate: bool = False) -> torch.Tensor:
        cm = cls._select_cm(accumulate)
        tp = torch.diag(cm)
        return tp / (cm.sum(dim=1) + cls.epsilon)

    @classmethod
    def iou_miou_score(cls, accumulate: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        cm = cls._select_cm(accumulate)
        tp = torch.diag(cm)
        each_class_counts = cm.sum(dim=1) + cm.sum(dim=0) - tp
        iou = tp / (each_class_counts + cls.epsilon)
        miou = torch.mean(iou)
        return iou, miou

    @classmethod
    def f_score(cls, accumulate: bool = False, beta: int = 1) -> torch.Tensor:
        cm = cls._select_cm(accumulate)
        tp = torch.diag(cm)
        precision = tp / (cm.sum(dim=0) + cls.epsilon)
        recall = tp / (cm.sum(dim=1) + cls.epsilon)
        beta_sq = beta ** 2
        return ((1 + beta_sq) * precision * recall /
                (precision * beta_sq + recall + cls.epsilon))
