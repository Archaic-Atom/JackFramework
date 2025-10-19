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
        self.conv_2d_layer_2 = Layer.conv_2d_layer(out_channels, out_channels, kernel_size,
                                                   padding=padding, dilation=dilation, norm=norm)
        self.downsample = downsample
        self.act_layer = act()

    # noinspection PyCallingNonCallable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv_2d_layer_1(x)
        x = self.conv_2d_layer_2(x)

        if self.downsample is not None:
            # Align the shortcut with the main path when stride or channels change.
            identity = self.downsample(identity)

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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv_2d_layer_1(x)
        x = self.conv_2d_layer_2(x)
        x = self.conv_2d_layer_3(x)

        if self.downsample is not None:
            # Keep residual tensor compatible with the bottleneck output.
            identity = self.downsample(identity)

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
            out_channels, out_channels, kernel_size, norm=norm)
        self.downsample = downsample
        self.act_layer = act()

    # noinspection PyCallingNonCallable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv_3d_layer_1(x)
        x = self.conv_3d_layer_2(x)

        if self.downsample is not None:
            # Downsample only the residual branch to preserve reference features.
            identity = self.downsample(identity)

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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv_3d_layer_1(x)
        x = self.conv_3d_layer_2(x)
        x = self.conv_3d_layer_3(x)

        if self.downsample is not None:
            # Keep residual tensor compatible with the bottleneck output.
            identity = self.downsample(identity)

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

        # Build atrous branches dynamically to avoid copy-paste maintenance errors.
        self.blocks = nn.ModuleList()
        self.blocks.append(Layer.conv_2d_layer(in_channels, out_channels, 1, padding=0,
                                               dilation=1, norm=norm, act=act))
        for rate in dilation:
            self.blocks.append(Layer.conv_2d_layer(
                in_channels, out_channels, 3, padding=rate,
                dilation=rate, norm=norm, act=act))

    # noinspection PyCallingNonCallable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Aggregate multi-scale context features from each atrous branch.
        branches = [block(x) for block in self.blocks]
        return torch.cat(branches, dim=1)


class SPPBlock(nn.Module):
    """docstring for SPPBlock"""

    def __init__(self, in_channels: int, out_channels: int,
                 act: object = NormActLayer.act_layer,
                 norm: object = NormActLayer.norm_layer) -> None:
        super().__init__()
        self.act = act
        self.norm = norm
        self.pool_sizes = (64, 32, 16, 8)
        # Pre-build pooling branches so forward can simply iterate over them.
        self.branches = nn.ModuleList(
            self.__make_block(in_channels, out_channels, pool_size)
            for pool_size in self.pool_sizes
        )

    def __make_block(self, in_channels: int, out_channels: int, ave_pool_size: int) -> nn.Sequential:
        layer = [
            nn.AvgPool2d((ave_pool_size, ave_pool_size), stride=(ave_pool_size, ave_pool_size),),
            Layer.conv_2d_layer(in_channels, out_channels, kernel_size=1,
                                padding=0, norm=self.norm, act=self.act)
        ]
        return nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.size()[2:]
        branches = []
        for branch in self.branches:
            pooled = branch(x)
            # Upsample pooled features back to the original spatial resolution.
            branches.append(F.interpolate(pooled, size=target_size, mode='bilinear', align_corners=False))

        return torch.cat(branches, dim=1)
