# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

try:
    from .block import Res2DBlock, Bottleneck2DBlock, Bottleneck3DBlock, Res3DBlock
    from .block import ASPPBlock, SPPBlock
except ImportError:
    from block import Res2DBlock, Bottleneck2DBlock, Bottleneck3DBlock, Res3DBlock
    from block import ASPPBlock, SPPBlock


class SysBasicUnitTestFramework(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _show_reslut(func: object, x: torch.tensor) -> None:
        out = func(x)
        print(out.shape)

    @staticmethod
    def _create_block() -> list:
        act, act1, norm = nn.SELU, nn.ReLU6, nn.BatchNorm2d
        block1 = Res2DBlock(64, 64, act=act, norm=norm)
        block2 = Bottleneck2DBlock(64, 64, act=act1, norm=norm)

        block3 = Res3DBlock(64, 64, act=act)
        block4 = Bottleneck3DBlock(64, 64, act=act)

        block5 = ASPPBlock(64, 64, act=act, norm=norm)
        block6 = SPPBlock(64, 64, act=act, norm=norm)
        return [block1, block2, block5, block6], [block3, block4]

    def _test_block(self) -> None:
        x, x1 = torch.ones((1, 64, 256, 256)), torch.ones((1, 64, 64, 64, 64))
        blocks_2d, blocks_3d = self._create_block()
        for item in blocks_2d:
            self._show_reslut(item, x)
        for item in blocks_3d:
            self._show_reslut(item, x1)

    def test(self) -> None:
        self._test_block()


def main() -> None:
    test_framework = SysBasicUnitTestFramework()
    test_framework.test()


if __name__ == '__main__':
    main()
