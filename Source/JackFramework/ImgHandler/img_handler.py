# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


class ImgHandler(object):
    """docstring for ImgHandler"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def img_tensor(imgs: list) -> list:
        imgs = list(map(lambda img: torch.Tensor(\
                        img.transpose(2,0,1)), imgs))
        return imgs

    @staticmethod
    def SSIM(x: torch.tensor, y: torch.tensor) -> torch.tensor:
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

