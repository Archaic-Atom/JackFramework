# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

from typing import TypeVar, Generic

tensor = TypeVar('tensor')

ACC_EPSILON = 1e-9


class Accuracy(object):
    """docstring for """

    def __init__(self):
        super().__init__()

    @staticmethod
    def d_1(res: tensor, gt: tensor, start_threshold: int = 2,
            threshold_num: int = 4, relted_error: float = 0.05,
            invaild_value: int = 0, max_disp: int = 192) -> tensor:
        mask = (gt != invaild_value) & (gt < max_disp)
        mask.detach_()
        acc_res = []
        with torch.no_grad():
            total_num = mask.int().sum()
            error = torch.abs(res[mask] - gt[mask])
            related_threshold = gt[mask] * relted_error
            for i in range(threshold_num):
                threshold = start_threshold + i
                acc = (error > threshold) & (error > related_threshold)
                acc_num = acc.int().sum()
                error_rate = acc_num / (total_num + ACC_EPSILON)
                acc_res.append(error_rate)
            mae = error.sum() / (total_num + ACC_EPSILON)
        return acc_res, mae

    @staticmethod
    def r2_score(res: tensor, gt: tensor) -> tensor:
        gt_mean = torch.mean(gt)
        ss_tot = torch.sum((gt - gt_mean) ** 2)
        ss_res = torch.sum((gt - res) ** 2)
        return 1 - ss_res / ss_tot

    @staticmethod
    def rmse_score(res: tensor, gt: tensor) -> tensor:
        return torch.sqrt(torch.mean((res - gt)**2))


def debug_main():
    pred = torch.rand(10, 600)
    gt = torch.rand(10, 600)
    out = Accuracy.rmse_score(pred, gt)
    print(out)


if __name__ == "__main__":
    debug_main()
