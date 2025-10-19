# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from JackFramework.Tools.tools import Tools

try:
    from ._meta_loss import MetaLoss
except ImportError:
    from _meta_loss import MetaLoss


class SegLoss(MetaLoss):
    """Loss helpers for segmentation tasks."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def focal_loss(res: torch.Tensor, gt: torch.Tensor, alpha: float = -1, gamma: float = 2,
                   reduction: str = 'mean', mode: str = 'bce') -> torch.Tensor:
        if res.shape[0] != gt.shape[0]:
            raise ValueError('Prediction and ground truth batch sizes must match for focal loss.')

        if mode == 'ce':
            gt = gt.long()
            log_t = F.cross_entropy(res, gt.squeeze(1), reduction='none')
        elif mode == 'bce':
            gt = gt.float()
            log_t = F.binary_cross_entropy_with_logits(res, gt, reduction="none")
        else:
            raise ValueError(f'Unsupported focal loss mode `{mode}`.')

        p_t = torch.exp(-log_t)
        loss = (1 - p_t) ** gamma * log_t

        if alpha >= 0:
            alpha_t = alpha * gt + (1 - alpha) * (1 - gt)
            loss = alpha_t * loss

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        elif reduction != 'none':
            raise ValueError(f'Unsupported reduction `{reduction}` for focal loss.')

        return loss

    @staticmethod
    def mutil_focal_loss(res: list, gt: torch.Tensor, alpha: float = -1, gamma: float = 2,
                         reduction: str = 'mean', mode: str = 'bce',
                         lambdas: list = None) -> torch.Tensor:
        length = len(res)
        if length == 0:
            raise ValueError('Prediction list for multi-scale focal loss must be non-empty.')
        _, _, h, w = gt.shape
        if lambdas is None:
            lambdas = [1.0] * length
        if len(lambdas) != length:
            raise ValueError('Length of `lambdas` must match number of prediction scales.')

        total_loss = 0
        for i in range(length):
            scale_factor = 1 / (2 ** i)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            scaled_target = F.interpolate(gt.float(), size=[new_h, new_w])
            _loss = SegLoss.focal_loss(res[i], scaled_target, alpha, gamma, reduction, mode)
            total_loss += _loss * lambdas[i]

        return total_loss

    @staticmethod
    def contrastive_loss(res: torch.Tensor, gt: torch.Tensor, margin: float) -> torch.Tensor:
        return torch.mean((1 - gt) * torch.pow(res, 2) +
                          gt * torch.pow(torch.clamp(margin - res, min=0), 2))

    @staticmethod
    def __dice_loss_func(res: torch.Tensor, gt: torch.Tensor, batch: int) -> torch.Tensor:
        res_vector = res.view(batch, -1)
        gt_vector = gt.view(batch, -1)
        intersection = (res_vector * gt_vector).sum(1)
        return 1 - torch.mean((2 * intersection) / (res_vector.sum(1)
                                                    + gt_vector.sum(1) + SegLoss.LOSS_EPSILON))

    @staticmethod
    def dice_loss(res: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        batch, num_classes, _, _ = res.shape
        if num_classes >= 2:
            res = F.softmax(res, dim=1)
            gt_one_hot = Tools.get_one_hot(gt, num_classes)
            loss = sum(SegLoss.__dice_loss_func(res[:, c], gt_one_hot[:, c], batch)
                       for c in range(num_classes))
            loss /= num_classes
        else:
            loss = SegLoss.__dice_loss_func(res, gt, batch)
        return loss
