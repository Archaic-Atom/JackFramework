from img_handler import ImgHandler 
from data_augmentation import DataAugmentation 
from img_io import ImgIO 
import numpy as np


def debug_main():
    path = 'Source/TestExample/DataAugSample.jpg'
    save_path = 'Source/TestExample/'
    img1 = ImgIO.read_img(path)
    img2 = ImgIO.read_img(path)
    imgs = []
    imgs.append(img1)
    imgs.append(img2)
    imgs = DataAugmentation.random_crop(imgs, 947, 432, 400, 400)
    # imgs = ImgHandler.img_tensor(imgs)
    print(imgs[0].shape, imgs[1].shape)
    ImgIO.write_img(save_path + 'Data_test1.png', np.array(imgs[0]))
    ImgIO.write_img(save_path + 'Data_test2.png', np.array(imgs[1]))


if __name__ == "__main__":
    debug_main()