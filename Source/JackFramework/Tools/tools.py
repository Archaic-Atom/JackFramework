# -*- coding: utf-8 -*-
import torch

class Tools(object):

    __TOOLS_INSTANCE = None

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__TOOLS_INSTANCE is None:
            cls.__TOOLS_INSTANCE = object.__new__(cls)
        return cls.__TOOLS_INSTANCE

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def __one_hot_func(label, num_classes):
        """
        :param label: label tensor with shape [H, W]
        :param num_classes: number of class
        :return: one-hot label tensor wish shape [num_claeese, H, W]
        """
        size = list(label.size())
        label = label.view(-1)
        ones = torch.sparse.torch.eye(num_classes)
        ones = ones.index_select(0, label.long())
        size.append(num_classes)
        ones = ones.view(*size)
        return ones.permute(2, 0, 1)

    @staticmethod
    def get_one_hot(label, num_classes):
        """
        :param label: label tensor with shape [B, 1, H, W]
        :param num_classes: number of class
        :return: one-hot label tensor wish shape [B, num_claeese, H, W]
        """

        batch, _, h, w = label.shape
        label_one_hot = torch.zeros([batch, num_classes, h, w], device=label.device)
        for b in range(batch):
            label_one_hot[b:b + 1] = Tools.__one_hot_func(label[b, 0], num_classes)

        return label_one_hot
