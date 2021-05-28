# -*- coding: utf-8 -*-
import os
import argparse

import JackFramework.SysBasic.define as sysdefine


# Parse the train model's para
class ArgsParser(object):
    """docstring for ArgsParser"""

    def __init__(self):
        super().__init__()

    def parse_args(self, info: str, user_define_func: object = None) -> object:
        parser = argparse.ArgumentParser(
            description="The deep learning framework (based on pytorch) - " + info)
        parser = self.__program_setting(parser)
        parser = self.__path_setting(parser)
        parser = self.__training_setting(parser)
        parser = self.__img_setting(parser)
        parser = self.__user_setting(parser)
        if user_define_func is not None:
            user_parser = user_define_func(parser)
            if type(user_parser) is type(parser):
                parser = user_parser

        args = parser.parse_args()
        return args

    @staticmethod
    def __program_setting(parser: object) -> object:
        parser.add_argument('--mode', default='train',
                            help='train or test')
        parser.add_argument('--gpu', type=int, default=sysdefine.GPU_NUM,
                            help='state the num of gpu: 0, 1, 2 or 3 ...')
        parser.add_argument('--auto_save_num', type=int,
                            default=sysdefine.AUTO_SAVE_NUM,
                            help='AUTO_SAVE_NUM')
        parser.add_argument('--dataloaderNum', type=int,
                            default=sysdefine.DATA_LOADER_NUM,
                            help='the number of dataloders')
        parser.add_argument('--pretrain', default=False,
                            type=ArgsParser.__str2bool,
                            help='true or false')
        parser.add_argument('--ip', default=sysdefine.IP,
                            help='ip')
        parser.add_argument('--port', default=sysdefine.PORT,
                            help='port')
        parser.add_argument('--dist', default=sysdefine.DIST,
                            type=ArgsParser.__str2bool,
                            help='distrobution')
        return parser

    @staticmethod
    def __path_setting(parser: object) -> object:
        parser.add_argument('--trainListPath', default=sysdefine.TRAIN_LIST_PATH,
                            help='training list path or testing list path')
        parser.add_argument('--valListPath', default=sysdefine.VAL_LIST_PATH,
                            help='val list path')
        parser.add_argument('--outputDir', default=sysdefine.DATA_OUTPUT_PATH,
                            help="The output's path. e.g. './result/'")
        parser.add_argument('--modelDir', default=sysdefine.MODEL_PATH,
                            help="The model's path. e.g. ./model/")
        parser.add_argument('--resultImgDir', default=sysdefine.RESULT_OUTPUT_PATH,
                            help="The test result img's path. e.g. ./ResultImg/")
        parser.add_argument('--log', default=sysdefine.LOG_OUTPUT_PATH,
                            help="the log file")
        return parser

    @staticmethod
    def __training_setting(parser: object) -> object:
        parser.add_argument('--sampleNum', type=int, default=sysdefine.SAMPLE_NUM,
                            help='the number of sample')
        # training setting
        parser.add_argument('--batchSize', type=int,
                            default=sysdefine.BATCH_SIZE,
                            help='Batch Size')
        parser.add_argument('--lr', default=sysdefine.LEARNING_RATE,
                            type=float,
                            help="Learning rate. e.g. 0.01, 0.001, 0.0001")
        parser.add_argument('--maxEpochs', default=sysdefine.MAX_EPOCHS,
                            type=int, help="Max step. e.g. 500")

        return parser

    @staticmethod
    def __img_setting(parser: object) -> object:
        parser.add_argument('--imgWidth', default=sysdefine.IMAGE_WIDTH, type=int,
                            help="Image's width. e.g. 512, In the training process is Clipped size")
        parser.add_argument('--imgHeight', default=sysdefine.IMAGE_HEIGHT, type=int,
                            help="Image's width. e.g. 256, In the training process is Clipped size")
        parser.add_argument('--imgNum', default=sysdefine.IMG_NUM, type=int,
                            help="The number of taining images")
        parser.add_argument('--valImgNum', default=sysdefine.VAL_IMG_NUM, type=int,
                            help="The number of val images")
        return parser

    @staticmethod
    def __user_setting(parser: object) -> object:
        parser.add_argument('--modelName', default=sysdefine.MODEL_NAME,
                            help='model name')
        parser.add_argument('--dataset', default=sysdefine.DATASET_NAME,
                            help="the dataset's name")
        return parser

    @staticmethod
    def __str2bool(arg: str) -> bool:
        if arg.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
