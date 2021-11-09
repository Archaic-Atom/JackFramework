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
    def random_crop(imgs: list, w: int, h: int,
                    crop_w: int, crop_h: int) -> list:
        x,y = DataAugmentation.random_org(w, h, crop_w, crop_h)
        imgs = list(map(lambda img: img[y:y + crop_h, \
                            x:x + crop_w, :], imgs))     
        return imgs
    
    @staticmethod
    def random_rotate(imgs: list, thro=0.5) -> list:
        if np.random.random() <= thro:
            rotote_k = np.random.randint(low=0, high=3)
            imgs = list(map(lambda img: np.rot90(img, rotote_k), imgs))
        return imgs
    
    @staticmethod
    def random_flip(imgs: list, thro=0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: np.flip(img, 0), imgs))
        if np.random.random() < thro:
            imgs = list(map(lambda img: np.flip(img, 1), imgs))
            return imgs



def debug_main():
    from PIL import Image
    img = Image.open('TestExample/DataAugSample.jpg')
    img = np.array(img)
    imgs = [img]
    img_crop = DataAugmentation.random_crop(imgs, 947, 432, 400, 400)
    img_rotate = DataAugmentation.random_rotate(imgs, thro=1)
    img_flip =  DataAugmentation.random_flip(imgs, 1)

    img_crop = Image.fromarray(img_crop[0])
    img_rotate = Image.fromarray(img_rotate[0])
    img_flip =  Image.fromarray(img_flip[0])

    img_crop = img_crop.save('TestExample/DataAug_crop.png')
    img_rotate = img_rotate.save('TestExample/DataAug_rotate.png')
    img_flip =  img_flip.save('TestExample/DataAug_flip.png')



if __name__ == "__main__":
    debug_main()
    






