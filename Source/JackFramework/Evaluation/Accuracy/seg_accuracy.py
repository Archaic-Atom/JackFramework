# -*- coding: utf-8 -*-
import torch

try:
    from ._meta_accuracy import MetaAccuracy
except ImportError:
    from _meta_accuracy import MetaAccuracy


class SegAccuracy(MetaAccuracy):
    """docstring for accuracy of segmentation task"""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def dice_score(res: torch.tensor, gt: torch.tensor) -> torch.tensor:
        intersection = (res * gt).sum()
        union = res.sum() + gt.sum() + SegAccuracy.ACC_EPSILON
        return 2 * intersection / union

    @staticmethod
    def generate_confusion_matrix(res: torch.tensor, gt: torch.tensor, num_classes: int) -> torch.tensor:
        res = res.flatten()
        gt = gt.flatten()
        mask = (gt >= 0) & (gt < num_classes)
        return torch.bincount(
            num_classes * gt[mask].int() + res[mask].int(),
            minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes)

    @staticmethod
    def precision_score(res: torch.tensor, gt: torch.tensor, num_classes: int) -> torch.tensor:
        cm = SegAccuracy.generate_confusion_matrix(res, gt, num_classes)
        tp = torch.diag(cm)
        return tp / (cm.sum(dim=0) + SegAccuracy.ACC_EPSILON)

    @staticmethod
    def recall_score(res: torch.tensor, gt: torch.tensor, num_classes: int) -> torch.tensor:
        cm = SegAccuracy.generate_confusion_matrix(res, gt, num_classes)
        tp = torch.diag(cm)
        return tp / (cm.sum(dim=1) + SegAccuracy.ACC_EPSILON)

    @staticmethod
    def pa_score(res: torch.tensor, gt: torch.tensor, num_classes: int) -> torch.tensor:
        cm = SegAccuracy.generate_confusion_matrix(res, gt, num_classes)
        tp = torch.diag(cm)
        return tp.sum() / (cm.sum() + SegAccuracy.ACC_EPSILON)

    @staticmethod
    def cpa_score(res: torch.tensor, gt: torch.tensor, num_classes: int) -> torch.tensor:
        cm = SegAccuracy.generate_confusion_matrix(res, gt, num_classes)
        tp = torch.diag(cm)
        sum_0 = cm.sum(dim=0)
        return tp / (sum_0 + SegAccuracy.ACC_EPSILON)

    @staticmethod
    def mpa_score(res: torch.tensor, gt: torch.tensor, num_classes: int) -> torch.tensor:
        cm = SegAccuracy.generate_confusion_matrix(res, gt, num_classes)
        tp = torch.diag(cm)
        sum_0 = cm.sum(dim=0)
        return torch.mean(tp / (sum_0 + SegAccuracy.ACC_EPSILON))

    @staticmethod
    def iou_miou_score(res: torch.tensor, gt: torch.tensor, num_classes: int) -> torch.tensor:
        cm = SegAccuracy.generate_confusion_matrix(res, gt, num_classes)
        tp = torch.diag(cm)
        each_class_counts = cm.sum(dim=1) + cm.sum(dim=0) - tp
        iou = tp / (each_class_counts + SegAccuracy.ACC_EPSILON)
        miou = torch.mean(iou)
        return iou, miou

    @staticmethod
    def fwiou_score(res: torch.tensor, gt: torch.tensor, num_classes: int) -> torch.tensor:
        cm = SegAccuracy.generate_confusion_matrix(res, gt, num_classes)
        tp = torch.diag(cm)
        sum_0 = cm.sum(dim=0)
        freq_weight = cm.sum(dim=1) / cm.sum()
        each_class_counts = cm.sum(dim=1) + sum_0 - tp
        return (tp * freq_weight / (each_class_counts + SegAccuracy.ACC_EPSILON)).sum()


def debug_main():
    from PIL import Image
    import torchvision.transforms as tfs

    img_path_list = ['TestExample/Accuracy_test1.png', 'TestExample/Accuracy_test2.png']
    trans = tfs.ToTensor()

    for i, img_path in enumerate(img_path_list):
        img = Image.open(img_path)
        img = trans(img)
        pred = img[:, 2:-2, 2:258].unsqueeze(0).cuda(i + 4)
        gt = img[:, 2:-2, 260:-2].unsqueeze(0).cuda(i + 4)

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


if __name__ == "__main__":
    debug_main()
