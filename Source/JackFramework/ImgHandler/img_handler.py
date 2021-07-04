# -*- coding: utf-8 -*-
import re
import random
import numpy as np
import sys
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImgHandler(object):
    """docstring for ImgHandler"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def read_img(path: str) -> np.array:
        return Image.open(path).convert("RGB")

    @staticmethod
    def read_single_channle_img(path: str) -> np.array:
        return Image.open(path)

    @staticmethod
    def read_pfm(filename: str) -> tuple:
        file = open(filename, 'rb')
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, scale

    @staticmethod
    def write_pfm(filename: str, image: np.array, scale: int = 1) -> None:
        with open(filename, mode='wb') as file:
            color = None

            if image.dtype.name != 'float32':
                raise Exception('Image dtype must be float32.')

            image = np.flipud(image)

            if len(image.shape) == 3 and image.shape[2] == 3:  # color image
                color = True
            elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
                color = False
            else:
                raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

            file.write(str.encode('PF\n' if color else 'Pf\n'))
            file.write(str.encode('%d %d\n' % (image.shape[1], image.shape[0])))

            endian = image.dtype.byteorder

            if endian == '<' or endian == '=' and sys.byteorder == 'little':
                scale = -scale

            file.write(str.encode('%f\n' % scale))

            image_string = image.tostring()
            file.write(image_string)

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

    @staticmethod
    def scale_pyramid(img: str, num_scales: int):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(
                nn.functional.interpolate(img,
                                          size=[nh, nw], mode='bilinear',
                                          align_corners=True)
            )
        return scaled_imgs

    @staticmethod
    def warp_img(img: torch.tensor, disp: torch.tensor) -> torch.tensor:
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                                                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                                                     width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        return F.grid_sample(img, 2 * flow_field - 1, mode='bilinear',
                               padding_mode='zeros')

    @staticmethod
    def generate_image_left(img: torch.tensor, disp: torch.tensor) -> torch.tensor:
        return ImgHandler.warp_img(img, -disp)

    @staticmethod
    def generate_image_right(img: torch.tensor, disp: torch.tensor) -> torch.tensor:
        return ImgHandler.warp_img(img, disp)

    @staticmethod
    def disp_smoothness(disp: torch.tensor, pyramid: torch.tensor, num_scales: int) -> list:
        disp_gradients_x = [ImgHandler.gradient_x(d) for d in disp]
        disp_gradients_y = [ImgHandler.gradient_y(d) for d in disp]

        image_gradients_x = [ImgHandler.gradient_x(img) for img in pyramid]
        image_gradients_y = [ImgHandler.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1,
                                           keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1,
                                           keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]
                        for i in range(num_scales)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]
                        for i in range(num_scales)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(num_scales)]

    @staticmethod
    def gradient_x(img: torch.tensor) -> torch.tensor:
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    @staticmethod
    def gradient_y(img: torch.tensor) -> torch.tensor:
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        return img[:, :, :-1, :] - img[:, :, 1:, :]
