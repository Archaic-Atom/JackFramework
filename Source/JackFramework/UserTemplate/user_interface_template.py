# -*- coding: utf-8 -*-
import argparse
from abc import ABCMeta, abstractmethod


class NetWorkInferenceTemplate(object):
    __metaclass__ = ABCMeta
    __NETWORK_INFERENCE = None

    def __init__(self):
        pass

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__NETWORK_INFERENCE is None:
            cls.__NETWORK_INFERENCE = object.__new__(cls)
        return cls.__NETWORK_INFERENCE

    @abstractmethod
    def inference(self, args: object) -> object:
        # get model and dataloader
        # return model, dataloader
        pass

    @abstractmethod
    def user_parser(self, parser: object) -> object:
        # parser.add_argument('--phase', default='train', help='train or test')
        # return parser
        pass

    @staticmethod
    def _str2bool(arg: str) -> bool:
        if arg.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
