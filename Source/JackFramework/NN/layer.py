# -*- coding: utf-8 -*-
import torch.nn as nn
from .ops import Ops


class NormActLayer(object):
    __NORM_LAYER_FUNC = None
    __ACT_LATER_FUNC = None
    __NORM_ACT_LAYER = None

    def __init__(self):
        super().__init__()

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__NORM_ACT_LAYER is None:
            cls.__NORM_ACT_LAYER = object.__new__(cls)
        return cls.__NORM_ACT_LAYER

    @staticmethod
    def set_norm_layer_func(norm_func: object) -> None:
        NormActLayer.__NORM_LAYER_FUNC = norm_func

    @staticmethod
    def set_act_layer_func(act_func: object) -> None:
        NormActLayer.__ACT_LATER_FUNC = act_func

    @staticmethod
    def act_layer() -> object:
        if NormActLayer.__ACT_LATER_FUNC is not None:
            return NormActLayer.__ACT_LATER_FUNC()
        return nn.ReLU(inplace=True)

    @staticmethod
    def norm_layer(out_channels: int) -> object:
        if NormActLayer.__NORM_LAYER_FUNC is not None:
            return NormActLayer.__NORM_LAYER_FUNC(out_channels)
        return Ops.gn(8, out_channels)

    # noinspection PyCallingNonCallable
    @staticmethod
    def norm_act_layer(layer: list, out_channels: int,
                       norm: object = None, act: object = None) -> list:
        if norm:
            layer.append(norm(out_channels))

        if act:
            layer.append(act())

        return layer


class Layer(object):
    """docstring for ClassName"""
    __LAYER = None

    def __init__(self):
        super().__init__()

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__LAYER is None:
            cls.__LAYER = object.__new__(cls)
        return cls.__LAYER

    @staticmethod
    def conv_1d_layer(in_channels: int, out_channels: int, kernel_size: int,
                      stride: int = 1, padding: int = 1, dilation: int = 1,
                      bias: bool = False, norm: object = NormActLayer.norm_layer,
                      act: object = NormActLayer.act_layer) -> object:
        layer = [
            Ops.conv_1d(
                in_channels, out_channels, kernel_size, stride,
                padding, dilation, bias=bias,
            )
        ]
        layer = NormActLayer.norm_act_layer(layer, out_channels, norm, act)
        return nn.Sequential(*layer)

    @ staticmethod
    def conv_2d_layer(in_channels: int, out_channels: int, kernel_size: int,
                      stride: int = 1, padding: int = 1, dilation: int = 1,
                      bias: bool = False, norm: object = NormActLayer.norm_layer,
                      act: object = NormActLayer.act_layer) -> object:
        layer = [
            Ops.conv_2d(
                in_channels, out_channels, kernel_size, stride,
                padding, dilation, bias=bias,
            )
        ]
        layer = NormActLayer.norm_act_layer(layer, out_channels, norm, act)
        return nn.Sequential(*layer)

    @ staticmethod
    def deconv_2d_layer(in_channels: int, out_channels: int, kernel_size: int,
                        stride: int = 1, padding: int = 0, output_padding: int = 0,
                        bias: bool = False, norm: object = NormActLayer.norm_layer,
                        act: object = NormActLayer.act_layer) -> object:
        layer = [
            Ops.deconv_2d(
                in_channels, out_channels, kernel_size, stride,
                padding, output_padding, bias=bias,
            )
        ]

        layer = NormActLayer.norm_act_layer(layer, out_channels, norm, act)
        return nn.Sequential(*layer)

    @ staticmethod
    def conv_3d_layer(in_channels: int, out_channels: int, kernel_size: int,
                      stride: int = 1, padding: int = 1, dilation: int = 1,
                      bias: bool = False, norm: object = NormActLayer.norm_layer,
                      act: object = NormActLayer.act_layer) -> object:
        layer = [
            Ops.conv_3d(
                in_channels, out_channels, kernel_size, stride,
                padding, dilation, bias=bias,
            )
        ]

        layer = NormActLayer.norm_act_layer(layer, out_channels, norm, act)
        return nn.Sequential(*layer)

    @ staticmethod
    def deconv_3d_layer(in_channels: int, out_channels: int, kernel_size: int,
                        stride: int = 1, padding: int = 0, output_padding: int = 0,
                        bias: bool = False, norm: object = NormActLayer.norm_layer,
                        act: object = NormActLayer.act_layer) -> object:
        layer = [
            Ops.deconv_3d(
                in_channels, out_channels, kernel_size, stride,
                padding, output_padding, bias=bias,
            )
        ]

        layer = NormActLayer.norm_act_layer(layer, out_channels, norm, act)
        return nn.Sequential(*layer)
