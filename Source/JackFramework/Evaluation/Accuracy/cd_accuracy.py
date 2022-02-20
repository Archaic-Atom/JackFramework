# -*- coding: utf-8 -*-
import torch

try:
    from ._meta_accuracy import MetaAccuracy
except ImportError:
    from _meta_accuracy import MetaAccuracy


class CDAccuracy(MetaAccuracy):
    """
    docstring for accuracy of change detection task

    If the key arg: accumulate is True, the functions will compute
    all scores by accumulated confusion matrix. Otherwise, by the
    confusion matrix of current mini-batch data.
    """
    __CONFUSION_MATRIX = None
    __TEMP_CONFUSION_MATRIX = None

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_cm(accumulate: bool = False):
        return CDAccuracy.__CONFUSION_MATRIX if accumulate else CDAccuracy.__TEMP_CONFUSION_MATRIX

    @staticmethod
    def reset_cm():
        CDAccuracy.__CONFUSION_MATRIX = None

    @staticmethod
    def generate_confusion_matrix(res: torch.tensor,
                                  gt: torch.tensor,
                                  num_classes: int) -> torch.tensor:
        res = res.flatten()
        gt = gt.flatten()
        mask = (gt >= 0) & (gt < num_classes)
        CDAccuracy.__TEMP_CONFUSION_MATRIX = torch.bincount(
            num_classes * gt[mask].int() + res[mask].int(),
            minlength=num_classes ** 2).reshape(num_classes, num_classes)
        print(CDAccuracy.__TEMP_CONFUSION_MATRIX)
        CDAccuracy.__CONFUSION_MATRIX += CDAccuracy.__TEMP_CONFUSION_MATRIX.cpu()

    @staticmethod
    def oa_score(accumulate: bool = False) -> torch.tensor:
        cm = CDAccuracy.get_cm(accumulate)
        tp = torch.diag(cm)
        return tp.sum() / (cm.sum() + CDAccuracy.ACC_EPSILON)

    @staticmethod
    def precision_score(accumulate: bool = False) -> torch.tensor:
        cm = CDAccuracy.get_cm(accumulate)
        tp = torch.diag(cm)
        return tp / (cm.sum(dim=0) + CDAccuracy.ACC_EPSILON)

    @staticmethod
    def recall_score(accumulate: bool = False) -> torch.tensor:
        cm = CDAccuracy.get_cm(accumulate)
        tp = torch.diag(cm)
        return tp / (cm.sum(dim=1) + CDAccuracy.ACC_EPSILON)

    @staticmethod
    def iou_miou_score(accumulate: bool = False) -> torch.tensor:
        cm = CDAccuracy.get_cm(accumulate)
        tp = torch.diag(cm)
        each_class_counts = cm.sum(dim=1) + cm.sum(dim=0) - tp
        iou = tp / (each_class_counts + CDAccuracy.ACC_EPSILON)
        miou = torch.mean(iou)
        return iou, miou

    @staticmethod
    def f_score(accumulate: bool = False,
                beta: int = 1) -> torch.tensor:
        cm = CDAccuracy.get_cm(accumulate)
        tp = torch.diag(cm)
        precision = tp / (cm.sum(dim=0) + CDAccuracy.ACC_EPSILON)
        recall = tp / (cm.sum(dim=1) + CDAccuracy.ACC_EPSILON)
        return ((1 + beta ** 2) * precision * recall / (precision * beta ** 2 + recall
                                                        + CDAccuracy.ACC_EPSILON))


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
        CDAccuracy.generate_confusion_matrix(pred[:1], gt[:1], 2)

        print('------------------change detection accuracy(no accumulation)------------------')
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

    print('------------------change detection accuracy(accumulation)------------------')
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


if __name__ == "__main__":
    debug_main()
