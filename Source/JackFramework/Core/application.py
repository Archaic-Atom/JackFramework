# -*- coding: UTF-8 -*-
"""Entry point orchestration for JackFramework applications."""

from collections.abc import Callable
from typing import Optional

import torch.multiprocessing as mp

from JackFramework.SysBasic.args_parser import ArgsParser
from JackFramework.SysBasic.init_handler import InitProgram
from JackFramework.SysBasic.log_handler import LogHandler as log

from .Mode import mode_selection


class Application(object):
    """Singleton wrapper coordinating argument parsing and mode execution."""

    __APPLICATION: Optional['Application'] = None

    def __new__(cls, *args: object, **kwargs: object) -> 'Application':
        if cls.__APPLICATION is None:
            cls.__APPLICATION = super().__new__(cls)
        return cls.__APPLICATION

    def __init__(self, user_interface: object, application_name: str = "") -> None:
        super().__init__()
        self.__user_interface = user_interface
        self.__application_name = application_name

    def set_user_interface(self, user_interface: object) -> None:
        self.__user_interface = user_interface

    def start(self) -> None:
        if self.__user_interface is None:
            raise RuntimeError('User interface has not been configured for the application.')

        args = self.__parse_args()
        if not InitProgram(args).init_program():
            log.error('Initialisation failed; aborting application start.')
            return

        mode_func = mode_selection(args, self.__user_interface.inference, args.mode)
        self._dist_app_start(mode_func, args.dist, args.gpu)
        log.info('The Application has finished successfully.')

    def __parse_args(self):
        parser_callback = getattr(self.__user_interface, 'user_parser', None)
        return ArgsParser().parse_args(self.__application_name, parser_callback)

    @staticmethod
    def _dist_app_start(mode_func: Callable, dist: bool, gpu_num: int) -> None:
        if dist:
            if gpu_num <= 0:
                raise ValueError('Distributed mode requires a positive `gpu` argument.')
            mp.spawn(mode_func, nprocs=gpu_num, join=True)
        else:
            mode_func()
