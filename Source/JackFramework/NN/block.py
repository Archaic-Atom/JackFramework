# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .layer import Layer, NormActLayer
except ImportError:
    from layer import Layer, NormActLayer


class Res2DBlock(nn.Module):
    """docstring for Res2DBlock"""

    # noinspection PyCallingNonCallable
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, dilation: int = 1, act: object = NormActLayer.act_layer,
                 norm: object = NormActLayer.norm_layer, downsample: object = None) -> None:
        super().__init__()
        self.conv_2d_layer_1 = Layer.conv_2d_layer(in_channels, out_channels, kernel_size,
                                                   stride, padding, dilation, norm=norm, act=act)
        self.conv_2d_layer_2 = Layer.conv_2d_layer(in_channels, out_channels, kernel_size,
                                                   padding=padding, dilation=dilation, norm=norm)
        self.downsample = downsample
        self.act_layer = act()

    # noinspection PyCallingNonCallable
    def forward(self, x: torch.tensor) -> torch.tensor:
        identity = x
        x = self.conv_2d_layer_1(x)
        x = self.conv_2d_layer_2(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        x += identity
        x = self.act_layer(x)
        return x


class Bottleneck2DBlock(nn.Module):
    """docstring for Bottleneck2DBlock"""

    # noinspection PyCallingNonCallable
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, act: object = NormActLayer.act_layer,
                 norm: object = NormActLayer.norm_layer, downsample: object = None) -> None:
        super().__init__()
        bottleneck_out_channels = out_channels // 4
        self.conv_2d_layer_1 = Layer.conv_2d_layer(
            in_channels, bottleneck_out_channels, 1, 1, 0, norm=norm, act=act)
        self.conv_2d_layer_2 = Layer.conv_2d_layer(
            bottleneck_out_channels, bottleneck_out_channels,
            kernel_size, stride, padding, norm=norm, act=act)
        self.conv_2d_layer_3 = Layer.conv_2d_layer(
            bottleneck_out_channels, out_channels, 1, 1, 0, norm=norm)
        self.downsample = downsample
        self.act_layer = act()

    # noinspection PyCallingNonCallable
    def forward(self, x: torch.tensor) -> torch.tensor:
        identity = x
        x = self.conv_2d_layer_1(x)
        x = self.conv_2d_layer_2(x)
        x = self.conv_2d_layer_3(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        x += identity
        x = self.act_layer(x)
        return x


class Res3DBlock(nn.Module):
    """docstring for Res3DBlock"""

    # noinspection PyCallingNonCallable
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, downsample: object = None, act: object = NormActLayer.act_layer,
                 norm: object = NormActLayer.norm_layer):
        super().__init__()
        self.conv_3d_layer_1 = Layer.conv_3d_layer(
            in_channels, out_channels, kernel_size, stride, padding,
            norm=norm, act=act)
        self.conv_3d_layer_2 = Layer.conv_3d_layer(
            in_channels, out_channels, kernel_size, norm=norm)
        self.downsample = downsample
        self.act_layer = act()

    # noinspection PyCallingNonCallable
    def forward(self, x: torch.tensor) -> torch.tensor:
        identity = x
        x = self.conv_3d_layer_1(x)
        x = self.conv_3d_layer_2(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        x += identity
        x = self.act_layer(x)
        return x


class Bottleneck3DBlock(nn.Module):
    """docstring for Bottleneck3DBlock"""

    # noinspection PyCallingNonCallable
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, downsample: object = None, act: object = NormActLayer.act_layer,
                 norm: object = NormActLayer.norm_layer) -> None:
        super().__init__()
        bottleneck_out_channels = out_channels // 4
        self.conv_3d_layer_1 = Layer.conv_3d_layer(
            in_channels, bottleneck_out_channels, 1, 1, 0,
            norm=norm, act=act)
        self.conv_3d_layer_2 = Layer.conv_3d_layer(
            bottleneck_out_channels, bottleneck_out_channels,
            kernel_size, stride, padding, norm=norm, act=act)
        self.conv_3d_layer_3 = Layer.conv_3d_layer(
            bottleneck_out_channels, out_channels, 1, 1, 0, norm=norm, act=False)
        self.downsample = downsample
        self.act_layer = act()

    # noinspection PyCallingNonCallable
    def forward(self, x: torch.tensor) -> torch.tensor:
        identity = x
        x = self.conv_3d_layer_1(x)
        x = self.conv_3d_layer_2(x)
        x = self.conv_3d_layer_3(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        x += identity
        x = self.act_layer(x)
        return x


class ASPPBlock(nn.Module):
    """docstring for ASPP"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 16,
                 act: object = NormActLayer.act_layer,
                 norm: object = NormActLayer.norm_layer) -> None:
        super().__init__()

        if stride == 8:
            dilation = [12, 24, 36]
        elif stride == 16:
            dilation = [6, 12, 18]
        else:
            raise NotImplementedError

        self.block_1 = Layer.conv_2d_layer(in_channels, out_channels, 1, padding=0,
                                           dilation=1, norm=norm, act=act)
        self.block_2 = Layer.conv_2d_layer(in_channels, out_channels, 3, padding=dilation[0],
                                           dilation=dilation[0], norm=norm, act=act)
        self.block_3 = Layer.conv_2d_layer(in_channels, out_channels, 3, padding=dilation[1],
                                           dilation=dilation[1], norm=norm, act=act)
        self.block_4 = Layer.conv_2d_layer(in_channels, out_channels, 3, padding=dilation[2],
                                           dilation=dilation[2], norm=norm, act=act)

    # noinspection PyCallingNonCallable
    def forward(self, x: torch.tensor) -> torch.tensor:
        branch_1 = self.block_1(x)
        branch_2 = self.block_2(x)
        branch_3 = self.block_3(x)
        branch_4 = self.block_4(x)
        x = torch.cat((branch_1, branch_2, branch_3, branch_4), dim=1)
        return x


class SPPBlock(nn.Module):
    """docstring for SPPBlock"""

    def __init__(self, in_channels: int, out_channels: int,
                 act: object = NormActLayer.act_layer,
                 norm: object = NormActLayer.norm_layer) -> None:
        super().__init__()
        self.act = act
        self.norm = norm
        self.branch_1 = self.__make_block(in_channels, out_channels, 64)
        self.branch_2 = self.__make_block(in_channels, out_channels, 32)
        self.branch_3 = self.__make_block(in_channels, out_channels, 16)
        self.branch_4 = self.__make_block(in_channels, out_channels, 8)

    def __make_block(self, in_channels: int, out_channels: int, ave_pool_size: int):
        layer = [
            nn.AvgPool2d((ave_pool_size, ave_pool_size), stride=(ave_pool_size, ave_pool_size),),
            Layer.conv_2d_layer(in_channels, out_channels, kernel_size=1,
                                padding=0, norm=self.norm, act=self.act)
        ]
        return nn.Sequential(*layer)

    def forward(self, x: torch.tensor) -> torch.tensor:
        branch_1 = self.branch_1(x)
        branch_1 = F.upsample(branch_1, (x.size()[2], x.size()[3]), mode='bilinear')

        branch_2 = self.branch_2(x)
        branch_2 = F.upsample(branch_2, (x.size()[2], x.size()[3]), mode='bilinear')

        branch_3 = self.branch_3(x)
        branch_3 = F.upsample(branch_3, (x.size()[2], x.size()[3]), mode='bilinear')

        branch_4 = self.branch_4(x)
        branch_4 = F.upsample(branch_4, (x.size()[2], x.size()[3]), mode='bilinear')

        x = torch.cat((branch_1, branch_2, branch_3, branch_4), dim=1)
        return x
