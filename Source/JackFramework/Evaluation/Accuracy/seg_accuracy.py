# -*- coding: utf-8 -*-
import torch

try:
    from ._meta_accuracy import MetaAccuracy
except ImportError:
    from _meta_accuracy import MetaAccuracy


class SegAccuracy(MetaAccuracy):
    """Segmentation metrics computed from confusion matrices."""

    @staticmethod
    def dice_score(res: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if res.shape != gt.shape:
            raise ValueError('Prediction and ground truth must share identical shapes for Dice score.')
        intersection = (res * gt).sum()
        union = res.sum() + gt.sum() + SegAccuracy.epsilon
        return 2 * intersection / union

    @staticmethod
    def generate_confusion_matrix(res: torch.Tensor,
                                  gt: torch.Tensor,
                                  num_classes: int) -> torch.Tensor:
        res = res.flatten().to(torch.int64)
        gt = gt.flatten().to(torch.int64)
        mask = (gt >= 0) & (gt < num_classes)
        return torch.bincount(
            num_classes * gt[mask] + res[mask],
            minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes).to(torch.float32)

    @staticmethod
    def precision_score(res: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
        cm = SegAccuracy.generate_confusion_matrix(res, gt, num_classes)
        tp = torch.diag(cm)
        return tp / (cm.sum(dim=0) + SegAccuracy.epsilon)

    @staticmethod
    def recall_score(res: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
        cm = SegAccuracy.generate_confusion_matrix(res, gt, num_classes)
        tp = torch.diag(cm)
        return tp / (cm.sum(dim=1) + SegAccuracy.epsilon)

    @staticmethod
    def pa_score(res: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
        cm = SegAccuracy.generate_confusion_matrix(res, gt, num_classes)
        tp = torch.diag(cm)
        return tp.sum() / (cm.sum() + SegAccuracy.epsilon)

    @staticmethod
    def cpa_score(res: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
        cm = SegAccuracy.generate_confusion_matrix(res, gt, num_classes)
        tp = torch.diag(cm)
        sum_0 = cm.sum(dim=0)
        return tp / (sum_0 + SegAccuracy.epsilon)

    @staticmethod
    def mpa_score(res: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
        cm = SegAccuracy.generate_confusion_matrix(res, gt, num_classes)
        tp = torch.diag(cm)
        sum_0 = cm.sum(dim=0)
        return torch.mean(tp / (sum_0 + SegAccuracy.epsilon))

    @staticmethod
    def iou_miou_score(res: torch.Tensor, gt: torch.Tensor, num_classes: int):
        cm = SegAccuracy.generate_confusion_matrix(res, gt, num_classes)
        tp = torch.diag(cm)
        each_class_counts = cm.sum(dim=1) + cm.sum(dim=0) - tp
        iou = tp / (each_class_counts + SegAccuracy.epsilon)
        miou = torch.mean(iou)
        return iou, miou

    @staticmethod
    def fwiou_score(res: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
        cm = SegAccuracy.generate_confusion_matrix(res, gt, num_classes)
        tp = torch.diag(cm)
        sum_0 = cm.sum(dim=0)
        freq_weight = cm.sum(dim=1) / (cm.sum() + SegAccuracy.epsilon)
        each_class_counts = cm.sum(dim=1) + sum_0 - tp
        return (tp * freq_weight / (each_class_counts + SegAccuracy.epsilon)).sum()
