# -*- coding: utf-8 -*-
import re
import numpy as np
import sys
from PIL import Image
import torch
import torch.nn as nn
import cv2

class ImgIO(object):
    """docstring for ImgIO"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def read_img(path: str, io_type=False) -> np.array:
        if io_type:
            img = cv2.imread(path)
        else:
            img = np.array(Image.open(path), np.float32)
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
        return img
    
    @staticmethod
    def write_img(path: str, img: np.array) -> None:
        img = np.array(img, np.uint8)
        if len(img.shape) == 2 or len(img.shape) == 3 \
                            and img.shape[2] == 3:
            img = Image.fromarray(img)
            img.save(path)
        else:
            raise Exception('Image must have H x W x 3, \
                        H x W x 1 or H x W dimensions.')

    @staticmethod
    def read_pfm(path: str) -> tuple:
        file = open(path, 'rb')
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
    def write_pfm(path: str, image: np.array, scale: int = 1) -> None:
        with open(path, mode='wb') as file:
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

