# -*- coding: utf-8 -*-
import numpy as np
import random

EPSILON = 1e-9


class DataAugmentation(object):
    """docstring for ClassName"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def random_org(w: int, h: int, crop_w: int, crop_h: int) -> tuple:
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)
        return x, y

    @staticmethod
    def standardize(img: object) -> object:
        """ normalize image input """
        img = img.astype(np.float32)
        var = np.var(img, axis=(0, 1), keepdims=True)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + EPSILON)

    @staticmethod
    def img_slice_3d(img: np.array, x: int, y: int,
                     w: int, h: int) -> np.array:
        return img[y:y + h, x:x + w, :]

    @staticmethod
    def img_slice_2d(img: np.array, x: int, y: int,
                     w: int, h: int) -> np.array:
        return img[y:y + h, x:x + w]
