# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from JackFramework.Tools.tools import Tools
# from tools import Tools


class Loss(object):
    """docstring for """

    __LOSS_INSTANCE = None
    LOSS_EPSILON = 1e-9

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__LOSS_INSTANCE is None:
            cls.__LOSS_INSTANCE = object.__new__(cls)
        return cls.__LOSS_INSTANCE

    def __init__(self):
        super().__init__()

    @staticmethod
    def smooth_l1(res: torch.tensor, gt: torch.tensor, mask_threshold_min: int,
                  mask_threshold_max: int) -> torch.tensor:
        mask = (gt > mask_threshold_min) & (gt < mask_threshold_max)
        mask.detach_()
        total_num = mask.int().sum() + Loss.LOSS_EPSILON
        return F.smooth_l1_loss(res[mask], gt[mask], reduction='sum') / total_num

    @staticmethod
    def focal_loss(res: torch.tensor, gt: torch.tensor, alpha: float = -1, gamma: float = 2,
                   reduction: str = 'mean', mode: str = 'bce') -> torch.tensor:
        if mode == 'ce':
            gt = gt.long()
            log_t = F.cross_entropy(res, gt.squeeze(1), reduction='none')
        elif mode == 'bce':
            gt = gt.float()
            log_t = F.binary_cross_entropy_with_logits(res, gt, reduction="none")

        p_t = torch.exp(-log_t)
        loss = (1 - p_t) ** gamma * log_t

        if alpha >= 0:
            alpha_t = alpha * gt + (1 - alpha) * (1 - gt)
            loss = alpha_t * loss

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    @ staticmethod
    def mutil_focal_loss(res: list, gt: torch.tensor, alpha: float = -1, gamma: float = 2,
                         reduction: str = 'mean', mode: str = 'bce',
                         lambdas: list = None) -> torch.tensor:
        loss, length = 0, len(res)
        _, _, h, w = gt.shape
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
    def contrastive_loss(res: torch.tensor, gt: torch.tensor, margin: float) -> torch.tensor:
        return torch.mean((1 - gt) * torch.pow(res, 2) +
                          gt * torch.pow(torch.clamp(margin - res, min=0), 2))

    @staticmethod
    def __dice_loss_func(res: torch.tensor, gt: torch.tensor, batch: int) -> torch.tensor:
        res_vector = res.view(batch, -1)
        gt_vector = gt.view(batch, -1)
        intersection = (res_vector * gt_vector).sum(1)
        return 1 - torch.mean((2 * intersection) / (res_vector.sum(1)
                              + gt_vector.sum(1) + Loss.LOSS_EPSILON))

    @staticmethod
    def dice_loss(res: torch.tensor, gt: torch.tensor) -> torch.tensor:
        batch, num_classes, _, _ = res.shape
        if num_classes >= 2:
            res = F.softmax(res, dim=1)
            gt_one_hot = Tools.get_one_hot(gt, num_classes)
            loss = sum(Loss.__dice_loss_func(res[:, c], gt_one_hot[:, c], batch)
                       for c in range(num_classes))
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

    pred = torch.cat((1 - pred1, pred1), dim=1)
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
