# -*- coding: utf-8 -*-
"""Common utility helpers used across JackFramework."""

from typing import Any, List

import torch
import torch.nn.functional as F

try:
    from collections.abc import Iterable as IterableType
except ImportError:  # pragma: no cover - compatibility with older Python versions
    from collections import Iterable as IterableType  # type: ignore


class Tools(object):
    """Singleton collection of lightweight utility helpers."""

    __TOOLS_INSTANCE = None

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__TOOLS_INSTANCE is None:
            cls.__TOOLS_INSTANCE = object.__new__(cls)
        return cls.__TOOLS_INSTANCE

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_one_hot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert integer segmentation labels to a dense one-hot tensor."""

        if label.dim() not in {3, 4}:
            raise ValueError('Label tensor must be 3D or 4D for one-hot conversion.')

        if label.dim() == 4:
            if label.size(1) != 1:
                raise ValueError('Expected label tensor with a single channel.')
            label = label.squeeze(1)

        label_long = label.long()
        one_hot = F.one_hot(label_long, num_classes=num_classes)  # shape: [B, H, W, C]
        one_hot = one_hot.permute(0, 3, 1, 2).contiguous()
        return one_hot.to(dtype=torch.float32, device=label.device)

    @staticmethod
    def convert2list(data_object: Any) -> List[Any]:
        """Ensure an object is represented as a list without touching tensors."""

        if isinstance(data_object, torch.Tensor):
            return [data_object]

        if isinstance(data_object, IterableType) and not isinstance(data_object, (str, bytes)):
            return list(data_object)

        return [data_object]


def debug_main() -> None:
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
