# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Ops(object):
    """docstring for ClassName"""
    __OPS = None

    def __init__(self):
        super().__init__()

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__OPS is None:
            cls.__OPS = object.__new__(cls)
        return cls.__OPS

    @staticmethod
    def conv_2d(in_channels: int, out_channels: int, kernel_size: int,
                stride: int = 1, padding: int = 0, dilation: int = 1,
                groups: int = 1, bias: bool = False,
                padding_mode: str = 'zeros') -> object:
        return nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups,
                         bias, padding_mode)

    @staticmethod
    def deconv_2d(in_channels: int, out_channels: int, kernel_size: int,
                  stride: int = 1, padding: int = 0, output_padding: int = 0,
                  groups: int = 1, dilation: int = 1, bias: bool = False,
                  padding_mode: str = 'zeros'):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                  stride, padding, output_padding, groups,
                                  bias, dilation, padding_mode)

    @staticmethod
    def conv_3d(in_channels: int, out_channels: int, kernel_size: int,
                stride: int = 1, padding: int = 0, dilation: int = 1,
                groups: int = 1, bias: bool = False,
                padding_mode: str = 'zeros') -> object:
        return nn.Conv3d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups,
                         bias, padding_mode)

    @staticmethod
    def deconv_3d(in_channels: int, out_channels: int, kernel_size: int,
                  stride: int = 1, padding: int = 0, output_padding: int = 0,
                  groups: int = 1, dilation: int = 1, bias: bool = False,
                  padding_mode: str = 'zeros') -> object:
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                  stride, padding, output_padding, groups,
                                  bias, dilation, padding_mode)

    @staticmethod
    def ave_pooling_2d(kernel_size: tuple, stride: tuple) -> object:
        return nn.AvgPool2d(kernel_size, stride)

    @staticmethod
    def bn(out_channels: int) -> object:
        return nn.BatchNorm2d(out_channels)

    @staticmethod
    def gn(group_num: int, out_channels: int) -> object:
        return nn.GroupNorm(group_num, out_channels)
