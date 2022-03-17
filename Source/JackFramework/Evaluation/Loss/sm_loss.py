# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

try:
    from ._meta_loss import MetaLoss
except ImportError:
    from _meta_loss import MetaLoss


class SMLoss(MetaLoss):
    """docstring for """

    def __init__(self):
        super().__init__()

    @staticmethod
    def smooth_l1(res: torch.tensor, gt: torch.tensor, mask_threshold_min: int,
                  mask_threshold_max: int) -> torch.tensor:
        mask = (gt > mask_threshold_min) & (gt < mask_threshold_max)
        mask.detach_()
        total_num = mask.int().sum() + SMLoss.LOSS_EPSILON
        return F.smooth_l1_loss(res[mask], gt[mask], reduction='sum') / total_num
