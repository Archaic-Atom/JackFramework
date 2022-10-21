# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np

try:
    from .data_augmentation import DataAugmentation
except ImportError:
    from data_augmentation import DataAugmentation


class DataAugmentationUnitTestFramework(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def open_img(path: str) -> list:
        img = Image.open(path)
        img = np.array(img)
        img = np.expand_dims(img, axis=3)
        return [img]

    @staticmethod
    def _save_img(path: str, img: np.array) -> None:
        img_data = Image.fromarray(img)
        img_data.save(path)

    def _test_data_aug(self) -> None:
        imgs = self.open_img('Example/DataAug/DataAugSample.jpg')

        # img_crop = DataAugmentation.random_crop(imgs, 947, 432, 400, 400)
        # img_rotate = DataAugmentation.random_rotate(imgs, thro=1)
        # img_flip = DataAugmentation.random_flip(imgs, 1)
        img_vertical_flip = DataAugmentation.random_vertical_flip(imgs, 1)

        # self._save_img('Example/DataAug/DataAug_crop.png', img_crop[0])
        # self._save_img('Example/DataAug/DataAug_rotate.png', img_rotate[0])
        # self._save_img('Example/DataAug/DataAug_flip.png', img_flip[0])
        self._save_img('Example/DataAug/DataAug_vflip.png', img_vertical_flip[0][:, :, :, 0])

    def test(self) -> None:
        self._test_data_aug()


def main() -> None:
    test_framework = DataAugmentationUnitTestFramework()
    test_framework.test()


if __name__ == '__main__':
    main()
