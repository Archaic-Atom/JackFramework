# -*- coding: utf-8 -*-
import os
import tifffile
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import cv2

import pandas as pd
from typing import TypeVar, Generic

from JackFramework.SysBasic.loghander import LogHandler as log
from JackFramework.ImgHandler import DataAugmentation
from JackFramework.SysBasic.switch import Switch
from JackFramework.ImgHandler.img_handler import ImgHandler
from JackFramework.FileHandler.filehandler import FileHandler


tensor = TypeVar('tensor')


class StereoDataset(Dataset):
    """docstring for DFCStereoDataset"""
    _DEPTH_DIVIDING = 256.0

    def __init__(self, args: object, list_path: str,
                 is_training: bool = False) -> None:
        self.__args = args
        self.__is_training = is_training
        self.__list_path = list_path

        input_dataframe = pd.read_csv(list_path)
        self.__left_img_path = input_dataframe["left_img"].values
        self.__right_img_path = input_dataframe["right_img"].values
        self.__gt_dsp_path = input_dataframe["gt_disp"].values

        self.__img_read_func, self.__label_read_func = \
            self.__read_func(args.dataset)

        if is_training:
            self.__get_path = self._get_training_path
            self.__data_steam = list(
                zip(self.__left_img_path,
                    self.__right_img_path, self.__gt_dsp_path))
        else:
            self.__get_path = self._get_testing_path
            self.__data_steam = list(
                zip(self.__left_img_path, self.__right_img_path))

    def save_kitti_test_data(self, img: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(
            args.resultImgDir, num)
        img = self._depth2img(img)
        self._save_png_img(path, img)

    def save_eth3d_test_data(self, img: np.array,
                             name: str, ttimes: str) -> None:
        args = self.__args
        path = args.resultImgDir + name + '.pfm'
        ImgHandler.write_pfm(path, img)
        path = args.resultImgDir + name + '.txt'
        with open(path, 'w') as f:
            f.write("runtime " + str(ttimes))
            f.close()

    def save_middlebury_test_data(self, img: np.array,
                                  name: str, ttimes: str) -> None:
        args = self.__args
        folder_name = args.resultImgDir + name + '/'
        FileHandler.mkdir(folder_name)
        method_name = "disp0" + args.modelName + "_RVC.pfm"
        path = folder_name + method_name
        ImgHandler.write_pfm(path, img)

        time_name = "time" + args.modelName + "_RVC.txt"
        path = folder_name + time_name
        with open(path, 'w') as f:
            f.write(str(ttimes))
            f.close()

    def crop_test_img(self, img: np.array,
                      top_pad: int, left_pad: int) -> np.array:
        if top_pad > 0 and left_pad > 0:
            img = img[top_pad:, : -left_pad]
        elif top_pad > 0:
            img = img[top_pad:, :]
        elif left_pad > 0:
            img = img[:, :-left_pad]
        return img

    def _generate_output_img_path(self, dir_path: str, num: str,
                                  filename_format: str = "%06d_10",
                                  img_type: str = ".png"):
        return dir_path + filename_format % num + img_type

    def _depth2img(self, img: np.array) -> np.array:
        img = np.array(img)
        img = (img * float(StereoDataset._DEPTH_DIVIDING)).astype(np.uint16)
        return img

        # save the png file
    def _save_png_img(self, path: str, img: np.array) -> None:
        cv2.imwrite(path, img)

    def __getitem__(self, idx: int):
        left_img_path, right_img_path, gt_dsp_path = self.__get_path(idx)
        return self._get_data(left_img_path, right_img_path, gt_dsp_path)

    def _get_training_path(self, idx: int) -> list:
        return self.__left_img_path[idx],\
            self.__right_img_path[idx], self.__gt_dsp_path[idx]

    def _get_testing_path(self, idx: int) -> list:
        return self.__left_img_path[idx], self.__right_img_path[idx], \
            self.__gt_dsp_path[idx]

    def _get_data(self, left_img_path, right_img_path, gt_dsp_path):
        if self.__is_training:
            return self._read_training_data(left_img_path,
                                            right_img_path, gt_dsp_path)
        return self._read_testing_data(left_img_path, right_img_path, gt_dsp_path)

    def _get_img_read_func(self):
        return self.__img_read_func, self.__label_read_func

    def _read_training_data(self, left_img_path: str,
                            right_img_path: str,
                            gt_dsp_path: str) -> object:
        args = self.__args

        width = args.imgWidth
        hight = args.imgHeight

        left_img = np.array(self.__img_read_func(left_img_path))
        right_img = np.array(self.__img_read_func(right_img_path))

        org_x, org_y = DataAugmentation.random_org(
            left_img.shape[1], left_img.shape[0], width, hight)
        left_img = DataAugmentation.img_slice_3d(
            left_img, org_x, org_y, width, hight)
        right_img = DataAugmentation.img_slice_3d(
            right_img, org_x, org_y, width, hight)

        left_img = DataAugmentation.standardize(left_img)
        right_img = DataAugmentation.standardize(right_img)

        gt_dsp = None
        if gt_dsp_path is not None:
            # print(gt_dsp_path)
            gt_dsp = np.array(self.__label_read_func(gt_dsp_path))
            gt_dsp = DataAugmentation.img_slice_2d(
                gt_dsp, org_x, org_y, width, hight)

        left_img = left_img.transpose(2, 0, 1)
        right_img = right_img.transpose(2, 0, 1)
        return left_img, right_img, gt_dsp

    @staticmethod
    def _padding_size(value: int, base: int = 64) -> int:
        off_set = 1
        times = value // base + off_set
        return times * base

    def _read_testing_data(self, left_img_path: str,
                           right_img_path: str,
                           gt_dsp_path: str) -> object:
        args = self.__args

        left_img = np.array(self.__img_read_func(left_img_path))
        right_img = np.array(self.__img_read_func(right_img_path))

        left_img = DataAugmentation.standardize(left_img)
        right_img = DataAugmentation.standardize(right_img)

        # pading size
        padding_height = self._padding_size(left_img.shape[0])
        padding_width = self._padding_size(left_img.shape[1])

        top_pad = padding_height - left_img.shape[0]
        left_pad = padding_width - right_img.shape[1]

        # pading
        left_img = np.lib.pad(left_img, ((
            top_pad, 0), (0, left_pad), (0, 0)),
            mode='constant', constant_values=0)
        right_img = np.lib.pad(right_img, ((
            top_pad, 0), (0, left_pad), (0, 0)),
            mode='constant', constant_values=0)

        left_img = left_img.transpose(2, 0, 1)
        right_img = right_img.transpose(2, 0, 1)

        gt_dsp = None
        if gt_dsp_path is not None:
            # print(gt_dsp_path)
            gt_dsp = np.array(self.__label_read_func(gt_dsp_path))
            gt_dsp = np.lib.pad(gt_dsp, ((
                top_pad, 0), (0, left_pad)),
                mode='constant', constant_values=0)

        name = ""
        if args.dataset in ["eth3d", "middlebury"]:
            off_set = 1
            pos = left_img_path.rfind('/')
            left_name = left_img_path[0:pos]
            pos = left_name.rfind('/')
            name = left_name[pos + off_set:]

        return left_img, right_img, gt_dsp, top_pad, left_pad, name

    def __len__(self):
        return len(self.__data_steam)

    def _read_png_disp(self, path: str) -> tensor:
        gt_dsp = ImgHandler.read_single_channle_img(path)
        gt_dsp = np.ascontiguousarray(
            gt_dsp, dtype=np.float32) / float(StereoDataset._DEPTH_DIVIDING)
        return gt_dsp

    def _read_pfm_disp(self, path: str) -> tensor:
        gt_dsp, _ = ImgHandler.read_pfm(path)
        return gt_dsp

    def __read_func(self, dataset_name: str) -> object:
        img_read_func = None
        label_read_func = None
        for case in Switch(dataset_name):
            if case('US3D'):
                img_read_func = tifffile.imread
                label_read_func = tifffile.imread
                break
            if case('kitti2012') or case('kitti2015'):
                img_read_func = ImgHandler.read_img
                label_read_func = self._read_png_disp
                break
            if case('eth3d') or case('middlebury') or case('sceneflow'):
                img_read_func = ImgHandler.read_img
                label_read_func = self._read_pfm_disp
                break
            if case():
                log.error("The model's name is error!!!")

        return img_read_func, label_read_func
