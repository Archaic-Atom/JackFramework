# -*- coding: UTF-8 -*-
from JackFramework.SysBasic.switch import Switch
from JackFramework.SysBasic.loghander import LogHandler as log

from .executor import Executor
from .background import BackGround


def mode_selection(args: object, inference: object, mode: str) -> object:
    mode_func = None
    for case in Switch(mode):
        if case('train'):
            log.info("Enter training mode")
            mode_func = Executor(args, inference, True).train
            break
        if case('test'):
            log.info("Enter testing mode")
            mode_func = Executor(args, inference, False).test
            break
        if case('background'):
            log.info("Enter background mode")
            mode_func = BackGround(args, inference, False).exec
            break
        if case('online'):
            log.info("Enter online mode")
            break
        if case('reinforcement_learning'):
            log.info("Enter reinforcement learning mode")
            break
        if case(''):
            log.error("The mode's name is error!!!")
    return mode_func
