# -*- coding: utf-8 -*-
import torch


class BaseAccuracy(object):
    """docstring for common accuracy"""
    __BASEACCURACY_INSTANCE = None
    ACC_EPSILON = 1e-9

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__BASEACCURACY_INSTANCE is None:
            cls.__BASEACCURACY_INSTANCE = object.__new__(cls)
        return cls.__BASEACCURACY_INSTANCE

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def rmse_score(res: torch.tensor, gt: torch.tensor) -> torch.tensor:
        res = res.view(res.size(0), -1)
        gt = gt.view(gt.size(0), -1)
        rmse = torch.sqrt(torch.sum((res - gt) ** 2, dim=1) / res.size(1))
        return torch.mean(rmse)


class SMAccuracy(object):
    """docstring for accuracy of stereo matching task"""
    __SMACCURACY_INSTANCE = None
    ACC_EPSILON = 1e-9

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__SMACCURACY_INSTANCE is None:
            cls.__SMACCURACY_INSTANCE = object.__new__(cls)
        return cls.__SMACCURACY_INSTANCE

    def __init__(self):
        super().__init__()

    @staticmethod
    def d_1(res: torch.tensor, gt: torch.tensor, start_threshold: int = 2,
            threshold_num: int = 4, relted_error: float = 0.05,
            invaild_value: int = 0, max_disp: int = 192) -> torch.tensor:
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
                error_rate = acc_num / (total_num + SMAccuracy.ACC_EPSILON)
                acc_res.append(error_rate)
            mae = error.sum() / (total_num + SMAccuracy.ACC_EPSILON)
        return acc_res, mae

    @staticmethod
    def r2_score(res: torch.tensor, gt: torch.tensor) -> torch.tensor:
        gt_mean = torch.mean(gt)
        ss_tot = torch.sum((gt - gt_mean) ** 2) + SMAccuracy.ACC_EPSILON
        ss_res = torch.sum((gt - res) ** 2)
        return ss_res / ss_tot

    @staticmethod
    def rmspe_score(res: torch.tensor, gt: torch.tensor) -> torch.tensor:
        return torch.sqrt(torch.mean((res - gt)**2 / gt))


class SegAccuracy(object):
    """docstring for accuracy of segmentation task"""
    __SEGACCURACY_INSTANCE = None
    ACC_EPSILON = 1e-9

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__SEGACCURACY_INSTANCE is None:
            cls.__SEGACCURACY_INSTANCE = object.__new__(cls)
        return cls.__SEGACCURACY_INSTANCE

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


class CDAccuracy(object):
    """
    docstring for accuracy of change detection task

    If the key arg: accumulate is True, the functions will caculate
    all scores by accumulated confusion matrix. Otherwise, by the
    confusion matrix of current mini-batch data.
    """
    __CDACCURACY_INSTANCE = None
    ACC_EPSILON = 1e-9
    __CONFUSION_MATRIX = 0
    __TEMP_CONFUSION_MATRIX = 0

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__CDACCURACY_INSTANCE is None:
            cls.__CDACCURACY_INSTANCE = object.__new__(cls)
        return cls.__CDACCURACY_INSTANCE

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_cm(accumulate: bool = False):
        if accumulate:
            return CDAccuracy.__CONFUSION_MATRIX
        else:
            return CDAccuracy.__TEMP_CONFUSION_MATRIX

    @staticmethod
    def reset_cm():
        CDAccuracy.__CONFUSION_MATRIX = 0

    @staticmethod
    def generate_confusion_matrix(res: torch.tensor,
                                  gt: torch.tensor,
                                  num_classes: int) -> torch.tensor:

        CDAccuracy.__TEMP_CONFUSION_MATRIX = 0
        res = res.flatten()
        gt = gt.flatten()
        mask = (gt >= 0) & (gt < num_classes)
        CDAccuracy.__TEMP_CONFUSION_MATRIX = torch.bincount(num_classes * gt[mask].int() + res[mask].int(),
                                                            minlength=num_classes**2).reshape(num_classes, num_classes)
        CDAccuracy.__CONFUSION_MATRIX += CDAccuracy.__TEMP_CONFUSION_MATRIX.cpu()

    @staticmethod
    def oa_score(accumulate: bool = False) -> torch.tensor:
        if accumulate:
            cm = CDAccuracy.__CONFUSION_MATRIX
        else:
            cm = CDAccuracy.__TEMP_CONFUSION_MATRIX
        tp = torch.diag(cm)
        return tp.sum() / (cm.sum() + CDAccuracy.ACC_EPSILON)

    @staticmethod
    def precision_score(accumulate: bool = False) -> torch.tensor:
        if accumulate:
            cm = CDAccuracy.__CONFUSION_MATRIX
        else:
            cm = CDAccuracy.__TEMP_CONFUSION_MATRIX
        tp = torch.diag(cm)
        return tp / (cm.sum(dim=0) + CDAccuracy.ACC_EPSILON)

    @staticmethod
    def recall_score(accumulate: bool = False) -> torch.tensor:
        if accumulate:
            cm = CDAccuracy.__CONFUSION_MATRIX
        else:
            cm = CDAccuracy.__TEMP_CONFUSION_MATRIX
        tp = torch.diag(cm)
        return tp / (cm.sum(dim=1) + CDAccuracy.ACC_EPSILON)

    @staticmethod
    def iou_miou_score(accumulate: bool = False) -> torch.tensor:
        if accumulate:
            cm = CDAccuracy.__CONFUSION_MATRIX
        else:
            cm = CDAccuracy.__TEMP_CONFUSION_MATRIX
        tp = torch.diag(cm)
        each_class_counts = cm.sum(dim=1) + cm.sum(dim=0) - tp
        iou = tp / (each_class_counts + CDAccuracy.ACC_EPSILON)
        miou = torch.mean(iou)
        return iou, miou

    @staticmethod
    def f_score(accumulate: bool = False,
                beta: int = 1) -> torch.tensor:
        if accumulate:
            cm = CDAccuracy.__CONFUSION_MATRIX
        else:
            cm = CDAccuracy.__TEMP_CONFUSION_MATRIX
        tp = torch.diag(cm)
        precision = tp / (cm.sum(dim=0) + CDAccuracy.ACC_EPSILON)
        recall = tp / (cm.sum(dim=1) + CDAccuracy.ACC_EPSILON)
        return (
            (1 + beta ** 2)
            * precision
            * recall
            / (precision * beta ** 2 + recall + CDAccuracy.ACC_EPSILON)
        )


def debug_main():

    from PIL import Image
    import torchvision.transforms as tfs

    # pred = torch.ones([10, 1, 10, 10])
    # gt = torch.ones([10, 1, 10, 10])

    # pred = torch.randint(low=0, high=10, size=[10, 1, 10, 10])
    # gt = torch.randint(low=0, high=10, size=[10, 1, 10, 10])
    # CDAccuracy.update_confusion_matrix(pred, gt, 10)

    img_path1 = 'TestExample/Accuracy_test1.png'
    img_path2 = 'TestExample/Accuracy_test2.png'
    img_path = [img_path1, img_path2]
    trans = tfs.ToTensor()
    for i in range(2):

        img = Image.open(img_path[i])
        img = trans(img)
        pred = img[:, 2:-2, 2:258].unsqueeze(0).cuda(i+4)
        gt = img[:, 2:-2, 260:-2].unsqueeze(0).cuda(i+4)

        CDAccuracy.generate_confusion_matrix(pred[0:1], gt[0:1], 2)

        print('------------------change detection accuracy(no accumulation)------------------')
        precision = CDAccuracy.precision_score()
        print('precision:', precision)
        recall = CDAccuracy.recall_score()
        print('recall:', recall)
        iou, miou = CDAccuracy.iou_miou_score()
        print('iou:', iou)
        print('miou:', miou)
        f1 = CDAccuracy.f_score(1)
        print('f1:', f1)
        oa = CDAccuracy.oa_score()
        print('oa:', oa)
        print(CDAccuracy.get_cm())

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

        print('------------------base accuracy------------------')
        rmse = BaseAccuracy.rmse_score(pred, gt)
        print('rmse:', rmse)

    print('------------------change detection accuracy(accumulation)------------------')
    precision = CDAccuracy.precision_score(True)
    print('precision:', precision)
    recall = CDAccuracy.recall_score(True)
    print('recall:', recall)
    iou, miou = CDAccuracy.iou_miou_score(True)
    print('iou:', iou)
    print('miou:', miou)
    f1 = CDAccuracy.f_score(True)
    print('f1:', f1)
    oa = CDAccuracy.oa_score(True)
    print('oa:', oa)
    print(CDAccuracy.get_cm(True))

    CDAccuracy.reset_cm()
    print(CDAccuracy.get_cm(True))


if __name__ == "__main__":
    debug_main()
