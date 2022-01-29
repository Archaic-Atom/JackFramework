# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class ImgHandler(object):
    """docstring for ImgHandler"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def hwc2cwh(imgs: list) -> list:
        imgs = list(map(lambda img: torch.Tensor(img.transpose(2, 0, 1)), imgs))
        return imgs

    @staticmethod
    def ssim(x: torch.tensor, y: torch.tensor) -> torch.tensor:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        ssim_n = (2 * mu_x_mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
        res = ssim_n / ssim_d

        return torch.clamp((1 - res) / 2, 0, 1)
