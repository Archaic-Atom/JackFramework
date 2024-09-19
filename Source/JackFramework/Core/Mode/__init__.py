# -*- coding: UTF-8 -*-
from collections.abc import Callable

from JackFramework.SysBasic.switch import Switch
from JackFramework.SysBasic.log_handler import LogHandler as log

from .test_proc import TestProc
from .train_proc import TrainProc
from .background import BackGround
from .web_proc import WebProc


def _get_mode_dict() -> dict:
    return {
        'train': TrainProc,
        'test': TestProc,
        'background': BackGround,
        'online': None,
        'reinforcement_learning': None,
        'web': WebProc,
    }


def mode_selection(args: object, user_inference_func: object, mode_name: str) -> Callable:
    mode_dict = _get_mode_dict()
    assert mode_name in mode_dict
    log.info(f'Enter {mode_name} mode')
    return mode_dict[mode_name](args, user_inference_func).exec
