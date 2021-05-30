# -*- coding: UTF-8 -*-
from JackFramework.SysBasic.inithandler import InitProgram
from JackFramework.Proc.executor import Executor
from JackFramework.SysBasic.argparser import ArgsParser
from JackFramework.SysBasic.loghander import LogHandler

import torch.multiprocessing as mp


class Application(object):
    """docstring for Application"""
    __APPLICATION = None

    def __init__(self, user_interface: object,
                 application_name: str = "") -> object:
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
        if not InitProgram(args).init_pro():
            return

        if args.dist:
            if args.mode == 'train':
                proc_func = Executor(args, self.__user_interface.inference, True).train
            else:
                proc_func = Executor(args, self.__user_interface.inference, False).test

            mp.spawn(proc_func, nprocs=args.gpu, join=True)

        else:
            if args.mode == 'train':
                Executor(args, self.__user_interface.inference, True).train()
            else:
                Executor(args, self.__user_interface.inference, False).test()

        LogHandler.info("The Application is finished!")
