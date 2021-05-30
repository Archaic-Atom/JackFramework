# -*- coding: utf-8 -*-
import re
import random
import numpy as np
import sys
from PIL import Image


class ImgHandler(object):
    """docstring for ImgHandler"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def read_img(path: str) -> np.array:
        img = Image.open(path).convert("RGB")
        return img

    @staticmethod
    def read_single_channle_img(path: str) -> np.array:
        img = Image.open(path)
        return img

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
        file = open(filename, mode='wb')
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

        file.close()
