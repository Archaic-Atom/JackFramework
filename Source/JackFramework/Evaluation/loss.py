# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

from typing import TypeVar, Generic

tensor = TypeVar('tensor')

LOSS_EPSILON = 1e-9


class Loss(object):
    """docstring for """

    def __init__(self):
        super().__init__()

    @ staticmethod
    def smooth_l1(res: tensor, gt: tensor,
                  mask_threshold_min: int, mask_threshold_max: int) -> tensor:
        mask = (gt > mask_threshold_min) & (gt < mask_threshold_max)
        mask.detach_()
        total_num = mask.int().sum() + LOSS_EPSILON
        return F.smooth_l1_loss(res[mask], gt[mask], reduction='sum') / total_num
