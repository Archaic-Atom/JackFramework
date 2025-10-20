# -*- coding: utf-8 -*-
"""Command-line argument handling for JackFramework programs."""

import argparse
from typing import Callable, Optional

import JackFramework.SysBasic.define as sys_define


class ArgsParser(object):
    """Aggregate framework defaults with optional user-provided arguments."""

    def __init__(self) -> None:
        super().__init__()

    def parse_args(self, info: str, user_define_func: Optional[Callable[[argparse.ArgumentParser],
                                                                          argparse.ArgumentParser]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=f"The deep learning framework (based on pytorch) - {info}"
        )
        parser = self.__program_setting(parser)
        parser = self.__path_setting(parser)
        parser = self.__training_setting(parser)
        parser = self.__img_setting(parser)
        parser = self.__user_setting(parser)
        parser = self.__load_user_define(parser, user_define_func)
        return parser.parse_args()

    @staticmethod
    def __load_user_define(parser: argparse.ArgumentParser,
                           user_define_func: Optional[Callable[[argparse.ArgumentParser],
                                                               argparse.ArgumentParser]]) -> argparse.ArgumentParser:
        if user_define_func is not None:
            user_parser = user_define_func(parser)
            if isinstance(user_parser, argparse.ArgumentParser):
                parser = user_parser
        return parser

    @staticmethod
    def __program_setting(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--mode', default='train', help='train or test')
        parser.add_argument('--gpu', type=int, default=sys_define.GPU_NUM,
                            help='number of GPUs to use: 0, 1, 2 ...')
        parser.add_argument('--auto_save_num', type=int, default=sys_define.AUTO_SAVE_NUM,
                            help='number of checkpoints to keep automatically')
        parser.add_argument('--dataloaderNum', type=int, default=sys_define.DATA_LOADER_NUM,
                            help='the number of DataLoader workers')
        parser.add_argument('--pretrain', default=False, type=ArgsParser.__str2bool,
                            help='load pretrained checkpoint if true')
        parser.add_argument('--ip', default=sys_define.IP, help='master address for distributed mode')
        parser.add_argument('--port', default=sys_define.PORT, help='master port for distributed mode')
        parser.add_argument('--dist', default=sys_define.DIST, type=ArgsParser.__str2bool,
                            help='enable distributed training')
        parser.add_argument('--nodes', type=int, default=sys_define.NODE_NUM,
                            help='number of nodes participating in distributed training')
        parser.add_argument('--node_rank', type=int, default=sys_define.NODE_RANK,
                            help='rank of the current node (0-based)')
        parser.add_argument('--debug', default=False, type=ArgsParser.__str2bool,
                            help='enable debug mode for verbose logging')
        parser.add_argument('--web_cmd', default='main.py runserver 0.0.0.0:8000',
                            help='Django management command to launch web UI')
        return parser

    @staticmethod
    def __path_setting(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--trainListPath', default=sys_define.TRAIN_LIST_PATH,
                            help='training list path or testing list path')
        parser.add_argument('--valListPath', default=sys_define.VAL_LIST_PATH,
                            help='validation list path')
        parser.add_argument('--outputDir', default=sys_define.DATA_OUTPUT_PATH,
                            help="output directory, e.g. './result/'")
        parser.add_argument('--modelDir', default=sys_define.MODEL_PATH,
                            help="model directory, e.g. './model/'")
        parser.add_argument('--resultImgDir', default=sys_define.RESULT_OUTPUT_PATH,
                            help="result image directory, e.g. './ResultImg/'")
        parser.add_argument('--log', default=sys_define.LOG_OUTPUT_PATH,
                            help='log directory')
        return parser

    @staticmethod
    def __training_setting(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--sampleNum', type=int, default=sys_define.SAMPLE_NUM,
                            help='the number of samples')
        parser.add_argument('--batchSize', type=int, default=sys_define.BATCH_SIZE,
                            help='batch size per iteration')
        parser.add_argument('--lr', type=float, default=sys_define.LEARNING_RATE,
                            help='learning rate, e.g. 0.01, 0.001, 0.0001')
        parser.add_argument('--maxEpochs', type=int, default=sys_define.MAX_EPOCHS,
                            help='maximum training epochs')
        return parser

    @staticmethod
    def __img_setting(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--imgWidth', type=int, default=sys_define.IMAGE_WIDTH,
                            help='image width used during training')
        parser.add_argument('--imgHeight', type=int, default=sys_define.IMAGE_HEIGHT,
                            help='image height used during training')
        parser.add_argument('--size_magnification', type=int, default=sys_define.SIZE_MAGNIFICATION,
                            help='feature map magnification factor')
        parser.add_argument('--imgNum', type=int, default=sys_define.IMG_NUM,
                            help='number of training images')
        parser.add_argument('--valImgNum', type=int, default=sys_define.VAL_IMG_NUM,
                            help='number of validation images')
        return parser

    @staticmethod
    def __user_setting(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--modelName', default=sys_define.MODEL_NAME,
                            help='model name')
        parser.add_argument('--dataset', default=sys_define.DATASET_NAME,
                            help='dataset name')
        return parser

    @staticmethod
    def __str2bool(arg: str) -> bool:
        lowered = arg.lower()
        if lowered in {'yes', 'true', 't', 'y', '1'}:
            return True
        if lowered in {'no', 'false', 'f', 'n', '0'}:
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')
