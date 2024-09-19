# -*- coding: utf-8 -*-
import collections
import torch

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections


class Tools(object):
    __TOOLS_INSTANCE = None

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__TOOLS_INSTANCE is None:
            cls.__TOOLS_INSTANCE = object.__new__(cls)
        return cls.__TOOLS_INSTANCE

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def __one_hot_func(label: torch.Tensor, num_classes: int) -> torch.Tensor:
        size = list(label.size())
        label = label.view(-1)
        ones = torch.sparse.torch.eye(num_classes)
        ones = ones.index_select(0, label.long())
        size.append(num_classes)
        ones = ones.view(*size)
        return ones.permute(2, 0, 1)

    @staticmethod
    def get_one_hot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
        off_set = 1
        batch, _, h, w = label.shape
        label_one_hot = torch.zeros([batch, num_classes, h, w], device=label.device)
        for b in range(batch):
            label_one_hot[b:b + off_set] = Tools.__one_hot_func(label[b, 0], num_classes)
        return label_one_hot

    @staticmethod
    def convert2list(data_object: any) -> list:
        if isinstance(data_object, collectionsAbc.Iterable)\
                and not isinstance(data_object, torch.Tensor):
            return list(data_object)
        return [data_object]


def debug_main():
    tools = Tools()
    # class object
    res = tools
    res = Tools.convert2list(res)
    print(res)
    # int
    res = 1
    res = Tools.convert2list(res)
    print(res)
    # tuple
    res = (1, 2, 3, 4)
    res = Tools.convert2list(res)
    print(res)


if __name__ == '__main__':
    debug_main()
