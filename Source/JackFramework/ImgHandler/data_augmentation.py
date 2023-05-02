# -*- coding: utf-8 -*-
import random
import numpy as np
import cv2


class DataAugmentation(object):
    """docstring for ClassName"""
    EPSILON = 1e-9

    def __init__(self):
        super().__init__()

    @staticmethod
    def random_org(w: int, h: int, crop_w: int, crop_h: int) -> tuple:
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)
        return x, y

    @staticmethod
    def standardize(img: object) -> object:
        img = img.astype(np.float32)
        var = np.var(img, axis=(0, 1), keepdims=True)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + DataAugmentation.EPSILON)

    @staticmethod
    def random_crop(imgs: list, w: int, h: int,
                    crop_w: int, crop_h: int) -> list:
        x, y = DataAugmentation.random_org(w, h, crop_w, crop_h)
        imgs = list(map(lambda img: img[y:y + crop_h,
                                        x:x + crop_w, :], imgs))
        return imgs

    @staticmethod
    def random_rotate(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() <= thro:
            rotate_k = np.random.randint(low=0, high=3)
            imgs = list(map(lambda img: np.rot90(img, rotate_k), imgs))
        return imgs

    @staticmethod
    def random_flip(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: np.flip(img, 0), imgs))
        if np.random.random() < thro:
            imgs = list(map(lambda img: np.flip(img, 1), imgs))
        return imgs

    @staticmethod
    def random_horizontal_flip(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: img[:, ::-1, ...], imgs))
        return imgs

    @staticmethod
    def random_vertical_flip(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: img[::-1, :, ...], imgs))
        return imgs

    @staticmethod
    def random_rotate90(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: img.swapaxes(1, 0)[:, ::-1, ...], imgs))
        return imgs

    @staticmethod
    def random_rotate180(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: img[:, ::-1, ...][::-1, :, ...], imgs))
        return imgs

    @staticmethod
    def random_rotate270(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: img.swapaxes(1, 0)[::-1, :, ...], imgs))
        return imgs

    @staticmethod
    def random_transpose(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: img.swapaxes(1, 0), imgs))
        return imgs

    @staticmethod
    def random_transverse(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: img[:, ::-1, ...].swapaxes(1, 0)[:, ::-1, ...], imgs))
        return imgs

    @staticmethod
    def random_scale(imgs: list, min_scale: float = 0.8, max_scale: float = 1.2) -> list:
        times = random.uniform(min_scale, max_scale)
        imgs = list(map(lambda img: cv2.resize(
            img, dsize=None, fx=times, fy=times, interpolation=cv2.INTER_LINEAR), imgs))
        return imgs
