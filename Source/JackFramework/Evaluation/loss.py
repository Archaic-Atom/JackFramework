# -*- coding: utf-8 -*-
from typing import Generic, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from JackFramework.ImgHandler.img_handler import ImgHandler
from JackFramework.Tools.tools import Tools

tensor = TypeVar('tensor')

LOSS_EPSILON = 1e-9


class Loss(object):
    """docstring for """

    __LOSS_INSTANCE = None

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__LOSS_INSTANCE is None:
            cls.__LOSS_INSTANCE = object.__new__(cls)
        return cls.__LOSS_INSTANCE

    def __init__(self):
        super().__init__()

    @staticmethod
    def smooth_l1(res: torch.tensor, gt: torch.tensor,
                  mask_threshold_min: int,
                  mask_threshold_max: int) -> torch.tensor:
        mask = (gt > mask_threshold_min) & (gt < mask_threshold_max)
        mask.detach_()
        total_num = mask.int().sum() + LOSS_EPSILON
        return F.smooth_l1_loss(res[mask], gt[mask], reduction='sum') / total_num

    @staticmethod
    def focal_loss(res: tensor, 
                gt: tensor, 
                alpha: float = -1, 
                gamma: float = 2,
                reduction: str = 'mean',
                mode: str = 'bce') -> tensor:
        """
        Params:
            res: Outputs of model with shape [B, C, H, W]
            gt: Labels of datasets with shape [B, C, H, W]
            alpha: Weighting factor in range (0, 1) to balance positive
                and negative examples. For example: 0.75 means  weighting 
                factor for positive examples.
            gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
            mode: 'ce' | 'bce'
                'ce': The channel num of res is 2, that means caculating 
                    focal_loss by using cross_entropy_loss 
                'bce': The channel num of res is 1, that means caculating 
                    focal_loss by using binary_cross_entropy_loss
        """
        if mode == 'ce':
            gt = gt.long()
            logp_t = F.cross_entropy(res, gt.squeeze(1), reduction='none')
        elif mode == 'bce':
            gt = gt.float()
            logp_t = F.binary_cross_entropy_with_logits(res, gt, reduction="none")

        p_t = torch.exp(-logp_t)
        loss = (1 - p_t) ** gamma * logp_t

        if alpha >= 0:
            alpha_t = alpha * gt + (1 - alpha) * (1 - gt)
            loss = alpha_t * loss

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    @ staticmethod
    def mutil_focal_loss(res: list, 
                gt: tensor, 
                alpha: float = -1, 
                gamma: float = 2,
                reduction: str = 'mean',
                mode: str = 'bce',
                lambdas: list = None) -> tensor:

        loss = 0
        length = len(res)
        _, _, h, w  = gt.shape
        if lambdas is None:
            lambdas = [1] * length

        for i in range(length):
            scale_factor = 1 / (2 ** i)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            scaled_target = F.interpolate(gt.float(), size=[new_h, new_w])
            _loss = Loss.focal_loss(res[i], scaled_target, alpha, gamma, reduction, mode)
            loss += _loss * lambdas[i]

        return loss

    @staticmethod
    def contrastive_loss(res: tensor, gt: tensor, margin: float) -> tensor:
        """
        :param res: Tensor with shape [B, C, H, W]
        :param gt: Tensor with shape [B, 1, H, W]
        :param margin: 
        :return: Average loss of batch data
        """
        return torch.mean((1- gt) * torch.pow(res, 2) +
                        gt * torch.pow(torch.clamp(margin - res, min=0), 2))

    @staticmethod
    def __dice_loss_func(res: tensor, gt: tensor, batch: int) -> tensor:
        res_vector = res.view(batch, -1)
        gt_vector = gt.view(batch, -1)
        intersection = (res_vector * gt_vector).sum()
        return 1 - torch.mean(
            (2 * intersection) / res_vector.sum() + gt_vector.sum() + LOSS_EPSILON
        )

    @staticmethod
    def dice_loss(res: tensor, gt: tensor) -> tensor:
        """
        :param res: Tensor with shape [B, C, H, W]
        :param gt: Tensor with shape [B, 1, H, W]
        :return: Average loss of batch data
        """
        batch, num_classes, _, _= res.shape
        if num_classes >= 2:
            res = F.softmax(res, dim=1)
            gt_one_hot = Tools.get_one_hot(gt, num_classes)
            loss = sum(
                Loss.__dice_loss_func(res[:, c], gt_one_hot[:, c], batch)
                for c in range(num_classes)
            )

            loss /= num_classes
        else:
            loss = Loss.__dice_loss_func(res, gt, batch)

        return loss



def debug_main():
    
    pred1 = torch.rand(size=[10, 1, 10, 10])
    pred2 = torch.rand(size=[10, 1, 10, 10])
    gt = torch.randint(low=0, high=2, size=[10, 1, 10, 10]).float()

    loss1 = Loss.focal_loss(pred1, gt, 0.75, 2, mode='bce')
    print(loss1)

    loss2 = Loss.mutil_focal_loss([pred1], gt, 0.75, 2, mode='bce')
    print(loss2)

    pred = torch.cat((1- pred1, pred1), dim=1)
    loss3 = Loss.focal_loss(pred, gt.long(), alpha=0.75, mode='ce')
    print(loss3)

    loss3 = Loss.mutil_focal_loss([pred], gt, 0.75, 2, mode='ce')
    print(loss3)

    pred = F.pairwise_distance(pred1, pred2, keepdim=True)
    loss4 = Loss.contrastive_loss(pred, gt, 2)
    print(loss4)

    pred3 = torch.rand(size=[10, 4, 10, 10])
    gt3 = torch.randint(low=0, high=2, size=[10, 1, 10, 10]).float()
    loss5 = Loss.dice_loss(pred3, gt3)
    print(loss5)

if __name__ == '__main__':
    debug_main()



