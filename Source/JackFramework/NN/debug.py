from typing import NoReturn
from block import *
from layer import *
import torch
import torch.nn as nn

def debug_main():
    act = nn.SELU
    act1 = nn.ReLU6
    norm = nn.BatchNorm2d
    block1 = Res2DBlock(64, 64, act=act, norm=norm)
    block2 = Bottleneck2DBlcok(64, 64, act=act1, norm=norm)

    block3 = Res3DBlock(64, 64, act=act)
    block4 = Bottleneck3DBlcok(64, 64, act=act)

    block5 = ASPPBlock(64, 64, act=act, norm=norm)
    block6 = SPPBlock(64, 64, act=act, norm=norm)

    x = torch.ones((1,64,256,256))
    x1 = torch.ones((1,64,64,64,64))
    x = block1(x)
    print(x.shape)
    x = block2(x)
    print(x.shape)
    x3 = block3(x1)
    print(x3.shape)
    x4 = block4(x1)
    print(x4.shape)
    x5 = block5(x)
    print(x5.shape)
    x6 = block6(x)
    print(x6.shape)
if __name__ == "__main__":
    debug_main()

