# -*- coding: UTF-8 -*-
"""Mode selection helpers for the Application entry point."""

from collections.abc import Callable
from typing import Dict, Type

from JackFramework.SysBasic.log_handler import LogHandler as log

from .background import BackGround
from .test_proc import TestProc
from .train_proc import TrainProc
from .web_proc import WebProc


MODE_REGISTRY: Dict[str, Type] = {
    'train': TrainProc,
    'test': TestProc,
    'background': BackGround,
    'web': WebProc,
}


def mode_selection(args: object, user_inference_func: object, mode_name: str) -> Callable:
    try:
        mode_cls = MODE_REGISTRY[mode_name]
    except KeyError as exc:
        available = ', '.join(sorted(MODE_REGISTRY))
        raise ValueError(f'Unknown mode `{mode_name}`. Available modes: {available}.') from exc

    log.info(f'Entering {mode_name} mode')
    return mode_cls(args, user_inference_func).exec
