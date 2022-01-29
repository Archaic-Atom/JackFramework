# -*- coding: utf-8 -*-
# sys setting
from JackFramework.SysBasic import define as sys_define

# custom lib
from JackFramework.SysBasic.loghander import LogHandler as log
from JackFramework.SysBasic.switch import Switch
from JackFramework.FileHandler.file_handler import FileHandler

# image handler
from JackFramework.ImgHandler.data_augmentation import DataAugmentation
from JackFramework.ImgHandler.img_io import ImgIO
from JackFramework.ImgHandler.img_handler import ImgHandler

# framework
from JackFramework.Core.application import Application
from JackFramework.SysBasic.result_str import ResultStr

# loss
from JackFramework.Evaluation.loss import Loss
from JackFramework.Evaluation.accuracy import BaseAccuracy, SMAccuracy, SegAccuracy, CDAccuracy

# nn
import JackFramework.NN as nn
import JackFramework.NN.block as block
from JackFramework.NN.layer import Layer as layer
from JackFramework.NN.layer import NormActLayer as norm_act_layer
from JackFramework.NN.ops import Ops as ops

# Contrib
import JackFramework.Contrib.Activation as act

# template
from JackFramework import UserTemplate

# Tools
from JackFramework.Tools.tools import Tools
from JackFramework.Tools.process_comm import NamedPipe


def version():
    return sys_define.VERSION
