# -*- coding: utf-8 -*-
"""Image IO helpers with improved validation and error handling."""

import re
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


class ImgIO(object):
    """Image read/write utilities supporting standard formats and PFM."""

    @staticmethod
    def read_img(path: str, use_cv: bool = False) -> np.ndarray:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f'Image file not found: {file_path}')

        if use_cv:
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f'OpenCV failed to read image: {file_path}')
            return img

        with Image.open(file_path) as pil_img:
            img = np.asarray(pil_img, dtype=np.float32)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        return img

    @staticmethod
    def write_img(path: str, img: np.ndarray) -> None:
        file_path = Path(path)
        img_array = np.asarray(img).squeeze()
        if img_array.ndim == 2:
            pil_img = Image.fromarray(img_array.astype(np.uint8))
        elif img_array.ndim == 3 and img_array.shape[2] in (1, 3):
            if img_array.shape[2] == 1:
                pil_img = Image.fromarray(img_array[:, :, 0].astype(np.uint8))
            else:
                pil_img = Image.fromarray(img_array.astype(np.uint8))
        else:
            raise ValueError('Image must have H x W x 3, H x W x 1, or H x W dimensions.')

        file_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(file_path)

    # ------------------------------------------------------------------
    @staticmethod
    def __get_color(header: str) -> bool:
        if header == 'PF':
            return True
        if header == 'Pf':
            return False
        raise ValueError('Not a PFM file.')

    @staticmethod
    def read_pfm(path: str) -> Tuple[np.ndarray, float]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f'PFM file not found: {file_path}')

        with file_path.open('rb') as file:
            header = file.readline().decode('utf-8').rstrip()
            color = ImgIO.__get_color(header)

            dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
            if not dim_match:
                raise ValueError('Malformed PFM header.')
            width, height = map(int, dim_match.groups())

            scale = float(file.readline().rstrip())
            endian = '<' if scale < 0 else '>'
            scale = abs(scale)

            data = np.fromfile(file, endian + 'f')

        shape = (height, width, 3) if color else (height, width)
        if data.size != np.prod(shape):
            raise ValueError('PFM data size does not match header dimensions.')

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, scale

    @staticmethod
    def write_pfm(path: str, image: np.ndarray, scale: int = 1) -> None:
        file_path = Path(path)
        image = np.asarray(image)
        if image.dtype.name != 'float32':
            raise ValueError('Image dtype must be float32 for PFM output.')

        image = np.flipud(image)
        if image.ndim == 3 and image.shape[2] == 3:
            color = True
        elif image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            color = False
        else:
            raise ValueError('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        with file_path.open('wb') as file:
            file.write(('PF\n' if color else 'Pf\n').encode('utf-8'))
            file.write(f'{image.shape[1]} {image.shape[0]}\n'.encode('utf-8'))

            endian = image.dtype.byteorder
            if endian == '<' or (endian == '=' and sys.byteorder == 'little'):
                scale = -scale

            file.write(f'{float(scale)}\n'.encode('utf-8'))
            file.write(image.tobytes())
