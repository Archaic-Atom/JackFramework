# -*- coding: utf-8 -*-
import torch

try:
    from ._meta_accuracy import MetaAccuracy
except ImportError:
    from _meta_accuracy import MetaAccuracy


class SMAccuracy(MetaAccuracy):
    """Stereo matching accuracy helpers."""

    @staticmethod
    def d_1(res: torch.Tensor, gt: torch.Tensor, start_threshold: int = 2,
            threshold_num: int = 4, related_error: float = 0.05,
            invalid_value: int = 0, max_disp: int = 192):
        mask = (gt != invalid_value) & (gt < max_disp)
        mask = mask.detach()
        error_rates = []
        with torch.no_grad():
            total_num = mask.int().sum()
            if total_num == 0:
                return [torch.tensor(0.0, device=res.device) for _ in range(threshold_num)], torch.tensor(0.0)
            error = torch.abs(res[mask] - gt[mask])
            related_threshold = gt[mask] * related_error
            for i in range(threshold_num):
                threshold = start_threshold + i
                exceed = (error > threshold) & (error > related_threshold)
                error_rate = exceed.int().sum() / (total_num + SMAccuracy.epsilon)
                error_rates.append(error_rate)
            mae = error.sum() / (total_num + SMAccuracy.epsilon)
        return error_rates, mae

    @staticmethod
    def r2_score(res: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        gt_mean = torch.mean(gt)
        ss_tot = torch.sum((gt - gt_mean) ** 2) + SMAccuracy.epsilon
        ss_res = torch.sum((gt - res) ** 2)
        return ss_res / ss_tot

    @staticmethod
    def rmspe_score(res: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean((res - gt) ** 2 / (gt + SMAccuracy.epsilon)))
