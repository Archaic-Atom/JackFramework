# -*- coding: UTF-8 -*-
from collections.abc import Callable
import torch.multiprocessing as mp

from JackFramework.SysBasic.inithandler import InitProgram
from JackFramework.SysBasic.argparser import ArgsParser
from JackFramework.SysBasic.loghander import LogHandler as log

from .Mode import mode_selection


class Application(object):
    """docstring for Application"""
    __APPLICATION = None

    def __init__(self, user_interface: object,
                 application_name: str = "") -> None:
        super().__init__()
        self.__user_interface = user_interface
        self.__application_name = application_name

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__APPLICATION is None:
            cls.__APPLICATION = object.__new__(cls)
        return cls.__APPLICATION

    def set_user_interface(self, user_interface: object) -> None:
        self.__user_interface = user_interface

    def start(self) -> None:
        args = ArgsParser().parse_args(self.__application_name,
                                       self.__user_interface.user_parser)
        if not InitProgram(args).init_program():
            return

        mode_func = mode_selection(args, self.__user_interface.inference, args.mode)
        self._dist_app_start(mode_func, args.dist, args.gpu)

        log.info("The Application is finished!")

    # noinspection PyCallingNonCallable
    @staticmethod
    def _dist_app_start(mode_func: Callable, dist: bool, gpu_num: int) -> None:
        if dist:
            mp.spawn(mode_func, nprocs=gpu_num, join=True)
        else:
            mode_func()
