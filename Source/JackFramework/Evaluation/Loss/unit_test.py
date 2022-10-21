# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

try:
    from .seg_loss import SegLoss
except ImportError:
    from seg_loss import SegLoss


class LossUnitTestFramework(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _test_seg_loss() -> None:
        print('------------------segmentation accuracy------------------')
        pred1, pred2 = torch.rand(size=[10, 1, 10, 10]), torch.rand(size=[10, 1, 10, 10])
        gt = torch.randint(low=0, high=2, size=[10, 1, 10, 10]).float()

        print('focal_loss: ', SegLoss.focal_loss(pred1, gt, 0.75, 2, mode='bce'))
        print('mutil_focal_loss: ', SegLoss.mutil_focal_loss([pred1], gt, 0.75, 2, mode='bce'))

        pred = torch.cat((1 - pred1, pred1), dim=1)
        print('focal_loss: ', SegLoss.focal_loss(pred, gt.long(), alpha=0.75, mode='ce'))
        print('mutil_focal_loss: ', SegLoss.mutil_focal_loss([pred], gt, 0.75, 2, mode='ce'))

        pred = F.pairwise_distance(pred1, pred2, keepdim=True)
        print('contrastive_loss:', SegLoss.contrastive_loss(pred, gt, 2))

        pred3 = torch.rand(size=[10, 4, 10, 10])
        gt3 = torch.randint(low=0, high=2, size=[10, 1, 10, 10]).float()
        print('dice_loss:', SegLoss.dice_loss(pred3, gt3))

    def test(self) -> None:
        self._test_seg_loss()


def main() -> None:
    test_framework = LossUnitTestFramework()
    test_framework.test()


if __name__ == '__main__':
    main()
