# -*- coding: utf-8 -*-
import torch
from ._meta_accuracy import MetaAccuracy


class BaseAccuracy(MetaAccuracy):
    """docstring for common accuracy"""
    __BASE_ACCURACY_INSTANCE = None

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__BASE_ACCURACY_INSTANCE is None:
            cls.__BASE_ACCURACY_INSTANCE = object.__new__(cls)
        return cls.__BASE_ACCURACY_INSTANCE

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def rmse_score(res: torch.tensor, gt: torch.tensor) -> torch.tensor:
        res = res.view(res.size(0), -1)
        gt = gt.view(gt.size(0), -1)
        rmse = torch.sqrt(torch.sum((res - gt) ** 2, dim=1) / res.size(1))
        return torch.mean(rmse)


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

        print('------------------base accuracy------------------')
        rmse = BaseAccuracy.rmse_score(pred, gt)
        print('rmse:', rmse)
