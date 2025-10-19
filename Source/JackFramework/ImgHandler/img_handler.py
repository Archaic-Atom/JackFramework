# -*- coding: utf-8 -*-
"""Tensor helpers for image preprocessing and similarity metrics."""

from typing import Iterable, List

import torch
import torch.nn as nn


class ImgHandler(object):
    @staticmethod
    def hwc2cwh(imgs: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        converted = []
        for img in imgs:
            if not isinstance(img, torch.Tensor):
                img = torch.as_tensor(img)
            if img.ndim != 3:
                raise ValueError('Expected images in HWC format (three dimensions).')
            converted.append(img.permute(2, 0, 1).contiguous())
        return converted

    @staticmethod
    def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError('SSIM expects input tensors with identical shapes.')

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        avg_pool = nn.AvgPool2d(3, 1, padding=1)
        mu_x = avg_pool(x)
        mu_y = avg_pool(y)
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = avg_pool(x * x) - mu_x_sq
        sigma_y = avg_pool(y * y) - mu_y_sq
        sigma_xy = avg_pool(x * y) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
        res = ssim_n / (ssim_d + 1e-12)
        return torch.clamp((1 - res) / 2, 0, 1)
