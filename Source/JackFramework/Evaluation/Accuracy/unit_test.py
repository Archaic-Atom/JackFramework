# -*- coding: utf-8 -*-
import torch
from PIL import Image
import torchvision.transforms as tfs

try:
    from .base_accuracy import BaseAccuracy
    from .cd_accuracy import CDAccuracy
    from .seg_accuracy import SegAccuracy
except ImportError:
    from base_accuracy import BaseAccuracy
    from cd_accuracy import CDAccuracy
    from seg_accuracy import SegAccuracy


class AccuracyUnitTestFramework(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _test_base_accuracy(pred: torch.Tensor, gt: torch.Tensor) -> None:
        print('------------------base accuracy------------------')
        rmse = BaseAccuracy.rmse_score(pred, gt)
        print('rmse:', rmse)

    @staticmethod
    def _test_cd_accuracy(pred: torch.Tensor, gt: torch.Tensor) -> None:
        CDAccuracy.generate_confusion_matrix(pred[:1], gt[:1], 2)
        print('---change detection accuracy(no accumulation)----')
        precision = CDAccuracy.precision_score()
        print('precision:', precision)
        recall = CDAccuracy.recall_score()
        print('recall:', recall)
        iou, miou = CDAccuracy.iou_miou_score()
        print('iou:', iou)
        print('miou:', miou)
        f1 = CDAccuracy.f_score(True)
        print('f1:', f1)
        oa = CDAccuracy.oa_score()
        print('oa:', oa)
        print(CDAccuracy.get_cm())

    @staticmethod
    def _test_cd_accuracy_acc() -> None:
        print('----change detection accuracy(accumulation)------')
        precision = CDAccuracy.precision_score(True)
        print('precision:', precision)
        recall = CDAccuracy.recall_score(True)
        print('recall:', recall)
        iou, miou = CDAccuracy.iou_miou_score(True)
        print('IoU:', iou)
        print('mIoU:', miou)
        f1 = CDAccuracy.f_score(True)
        print('f1:', f1)
        oa = CDAccuracy.oa_score(True)
        print('oa:', oa)
        print(CDAccuracy.get_cm(True))

        CDAccuracy.reset_cm()
        print(CDAccuracy.get_cm(True))

    @staticmethod
    def _test_seg_accuracy(pred: torch.Tensor, gt: torch.Tensor) -> None:
        print('------------------segmentation accuracy------------------')
        pa = SegAccuracy.pa_score(pred, gt, 2)
        print('pa:', pa)
        cpa = SegAccuracy.cpa_score(pred, gt, 2)
        print('cpa:', cpa)
        mpa = SegAccuracy.mpa_score(pred, gt, 2)
        print('mpa:', mpa)
        iou, miou = SegAccuracy.iou_miou_score(pred, gt, 2)
        print('iou:', iou)
        print('miou:', miou)
        fwiou = SegAccuracy.fwiou_score(pred, gt, 2)
        print('fwiou:', fwiou)
        dice = SegAccuracy.dice_score(pred, gt)
        print('dice:', dice)
        precision = SegAccuracy.precision_score(pred, gt, 2)
        print('precision:', precision)
        recall = SegAccuracy.recall_score(pred, gt, 2)
        print('recall:', recall)

    @staticmethod
    def read_img(path: str, idx: int) -> torch.Tensor:
        trans = tfs.ToTensor()
        img = trans(Image.open(path))
        pred = img[:, 2:-2, 2:258].unsqueeze(0).cuda(idx + 4)
        gt = img[:, 2:-2, 260:-2].unsqueeze(0).cuda(idx + 4)
        return pred, gt

    def test(self, img_path_list: list) -> None:
        for i, img_path in enumerate(img_path_list):
            pred, gt = self.read_img(img_path, i)
            self._test_base_accuracy(pred, gt)
            self._test_seg_accuracy(pred, gt)
            self._test_cd_accuracy(pred, gt)
        self._test_cd_accuracy_acc()


def main() -> None:
    img_path_list = ['Example/Accuracy/Accuracy_test1.png',
                     'Example/Accuracy/Accuracy_test2.png']
    test_framework = AccuracyUnitTestFramework()
    test_framework.test(img_path_list)


if __name__ == '__main__':
    main()
