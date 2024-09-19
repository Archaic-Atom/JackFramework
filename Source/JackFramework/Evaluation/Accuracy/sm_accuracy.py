# -*- coding: utf-8 -*-
import torch
try:
    from ._meta_accuracy import MetaAccuracy
except ImportError:
    from _meta_accuracy import MetaAccuracy


class SMAccuracy(MetaAccuracy):
    """docstring for accuracy of stereo matching task"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def d_1(res: torch.Tensor, gt: torch.Tensor, start_threshold: int = 2,
            threshold_num: int = 4, related_error: float = 0.05,
            invalid_value: int = 0, max_disp: int = 192) -> torch.Tensor:
        mask = (gt != invalid_value) & (gt < max_disp)
        mask.detach_()
        acc_res = []
        with torch.no_grad():
            total_num = mask.int().sum()
            error = torch.abs(res[mask] - gt[mask])
            related_threshold = gt[mask] * related_error
            for i in range(threshold_num):
                threshold = start_threshold + i
                acc = (error > threshold) & (error > related_threshold)
                acc_num = acc.int().sum()
                error_rate = acc_num / (total_num + SMAccuracy.ACC_EPSILON)
                acc_res.append(error_rate)
            mae = error.sum() / (total_num + SMAccuracy.ACC_EPSILON)
        return acc_res, mae

    @staticmethod
    def r2_score(res: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        gt_mean = torch.mean(gt)
        ss_tot = torch.sum((gt - gt_mean) ** 2) + SMAccuracy.ACC_EPSILON
        ss_res = torch.sum((gt - res) ** 2)
        return ss_res / ss_tot

    @staticmethod
    def rmspe_score(res: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean((res - gt) ** 2 / gt))
