# -*- coding: utf-8 -*-
"""Common numpy-based augmentation primitives used by JackFramework."""

import random
from typing import Callable, Iterable, List, Sequence

import cv2
import numpy as np


ArrayLike = np.ndarray


class DataAugmentation(object):
    """Utility helpers to apply synchronized random transforms across images."""

    EPSILON = 1e-9

    @staticmethod
    def _apply(imgs: Sequence[ArrayLike], op: Callable[[ArrayLike], ArrayLike]) -> List[ArrayLike]:
        return [op(img) for img in imgs]

    @staticmethod
    def _validate_dimensions(img: ArrayLike, min_dims: int = 2) -> None:
        if img.ndim < min_dims:
            raise ValueError('Input images must have at least 2 dimensions (H, W[, C]).')

    # ------------------------------------------------------------------
    @staticmethod
    def random_origin(width: int, height: int, crop_w: int, crop_h: int) -> tuple[int, int]:
        if crop_w > width or crop_h > height:
            raise ValueError('Crop size must be smaller than the source dimensions.')
        x = random.randint(0, width - crop_w)
        y = random.randint(0, height - crop_h)
        return x, y

    @staticmethod
    def standardize(img: ArrayLike) -> ArrayLike:
        img = img.astype(np.float32, copy=False)
        var = np.var(img, axis=(0, 1), keepdims=True)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + DataAugmentation.EPSILON)

    # ------------------------------------------------------------------
    @staticmethod
    def random_crop(imgs: Sequence[ArrayLike], crop_w: int, crop_h: int) -> List[ArrayLike]:
        if not imgs:
            return []
        h, w = imgs[0].shape[:2]
        x, y = DataAugmentation.random_origin(w, h, crop_w, crop_h)
        return DataAugmentation._apply(imgs, lambda img: img[y:y + crop_h, x:x + crop_w, ...])

    @staticmethod
    def random_flip(imgs: Sequence[ArrayLike], probability: float = 0.5) -> List[ArrayLike]:
        if np.random.random() < probability:
            imgs = DataAugmentation._apply(imgs, lambda img: np.flip(img, 0))
        if np.random.random() < probability:
            imgs = DataAugmentation._apply(imgs, lambda img: np.flip(img, 1))
        return imgs

    @staticmethod
    def random_horizontal_flip(imgs: Sequence[ArrayLike], probability: float = 0.5) -> List[ArrayLike]:
        if np.random.random() < probability:
            return DataAugmentation._apply(imgs, lambda img: img[:, ::-1, ...])
        return list(imgs)

    @staticmethod
    def random_vertical_flip(imgs: Sequence[ArrayLike], probability: float = 0.5) -> List[ArrayLike]:
        if np.random.random() < probability:
            return DataAugmentation._apply(imgs, lambda img: img[::-1, :, ...])
        return list(imgs)

    @staticmethod
    def random_rotate(imgs: Sequence[ArrayLike], probability: float = 0.5) -> List[ArrayLike]:
        if np.random.random() > probability:
            return list(imgs)
        rotate_k = np.random.randint(low=0, high=4)
        return DataAugmentation._apply(imgs, lambda img: np.rot90(img, rotate_k))

    @staticmethod
    def random_transpose(imgs: Sequence[ArrayLike], probability: float = 0.5) -> List[ArrayLike]:
        if np.random.random() < probability:
            return DataAugmentation._apply(imgs, lambda img: np.swapaxes(img, 0, 1))
        return list(imgs)

    @staticmethod
    def random_rotate90(imgs: Sequence[ArrayLike], probability: float = 0.5) -> List[ArrayLike]:
        if np.random.random() < probability:
            return DataAugmentation._apply(imgs, lambda img: np.rot90(img, 1))
        return list(imgs)

    @staticmethod
    def random_rotate180(imgs: Sequence[ArrayLike], probability: float = 0.5) -> List[ArrayLike]:
        if np.random.random() < probability:
            return DataAugmentation._apply(imgs, lambda img: np.rot90(img, 2))
        return list(imgs)

    @staticmethod
    def random_rotate270(imgs: Sequence[ArrayLike], probability: float = 0.5) -> List[ArrayLike]:
        if np.random.random() < probability:
            return DataAugmentation._apply(imgs, lambda img: np.rot90(img, 3))
        return list(imgs)

    @staticmethod
    def random_transverse(imgs: Sequence[ArrayLike], probability: float = 0.5) -> List[ArrayLike]:
        if np.random.random() < probability:
            return DataAugmentation._apply(imgs, lambda img: np.rot90(img[:, ::-1, ...], 3))
        return list(imgs)

    @staticmethod
    def random_scale(imgs: Sequence[ArrayLike], min_scale: float = 0.8, max_scale: float = 1.2) -> List[ArrayLike]:
        scale = random.uniform(min_scale, max_scale)
        return DataAugmentation._apply(
            imgs,
            lambda img: cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR),
        )
