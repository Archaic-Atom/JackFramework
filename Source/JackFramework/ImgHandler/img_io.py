# -*- coding: utf-8 -*-
import re
import numpy as np
import sys
from PIL import Image
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
        img = np.array(img, np.uint8).squeeze()
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
        color, scale, endian = None, None, None
        width, height = None, None

        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        if dim_match := re.match(
            r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8')
        ):
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
