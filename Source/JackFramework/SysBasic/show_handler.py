# -*- coding: utf-8 -*-
"""High-level helpers for displaying training progress and tensorboard stats."""

import time
from typing import Optional

from JackFramework.SysBasic.log_handler import LogHandler as log
from JackFramework.FileHandler.tensorboard_handler import TensorboardHandler

from ._show_manager import ShowManager
from .process_bar import ShowProcess


class ShowHandler(ShowManager):
    __ShowHandler = None
    __TENSORBOARD_HANDLER: Optional[TensorboardHandler] = None
    __PROCESS_BAR: Optional[ShowProcess] = None
    __START_TIME: Optional[float] = None
    __DURATION: Optional[float] = None
    __REST_TIME: Optional[float] = None

    def __init__(self) -> None:
        super().__init__()

    @ShowManager.show_method
    def __init_tensorboard_handler(self, args: object) -> None:
        self.init_tensorboard_handler(args)

    def reinit_log_tensorboard_handler(self, args: object) -> None:
        self.__reinit_log_handler(args)
        self.__init_tensorboard_handler(args)

    @staticmethod
    def init_show_setting(training_iteration: int, bar_info: str) -> None:
        ShowHandler.__PROCESS_BAR = ShowProcess(training_iteration, bar_info)
        ShowHandler.__START_TIME = time.time()
        ShowHandler.__DURATION = None
        ShowHandler.__REST_TIME = None

    @staticmethod
    def init_tensorboard_handler(args: object) -> None:
        ShowHandler.__TENSORBOARD_HANDLER = TensorboardHandler(args)

    @staticmethod
    def calculate_ave_runtime(total_iteration: int, training_iteration: int) -> None:
        if ShowHandler.__START_TIME is None:
            return
        elapsed = time.time() - ShowHandler.__START_TIME
        average_duration = elapsed / max(total_iteration, 1)
        remaining_iterations = max(training_iteration - total_iteration, 0)
        ShowHandler.__DURATION = average_duration
        ShowHandler.__REST_TIME = remaining_iterations * average_duration

    @staticmethod
    def stop_show_setting() -> None:
        if ShowHandler.__PROCESS_BAR is not None:
            ShowHandler.__PROCESS_BAR.close()
        ShowHandler.__PROCESS_BAR = None
        ShowHandler.__START_TIME = None
        ShowHandler.__DURATION = None
        ShowHandler.__REST_TIME = None

    @staticmethod
    def duration() -> Optional[float]:
        if ShowHandler.__START_TIME is None:
            return None
        return time.time() - ShowHandler.__START_TIME

    @staticmethod
    def __reinit_log_handler(args: object) -> None:
        if args.dist:
            log().init_log(args.outputDir, args.pretrain)
            log().info('LogHandler is reinitialized!')

    @staticmethod
    def update_show_bar(info_str: str) -> None:
        if ShowHandler.__PROCESS_BAR is None:
            return
        ShowHandler.__PROCESS_BAR.show_process(show_info=info_str,
                                               rest_time=ShowHandler.__REST_TIME,
                                               duration=ShowHandler.__DURATION)

    @staticmethod
    def write_tensorboard(epoch: int, ave_tower_loss: list,
                          ave_tower_acc: list, bar_info: str) -> None:
        if ShowHandler.__TENSORBOARD_HANDLER is None:
            return
        ShowHandler.__TENSORBOARD_HANDLER.write_data(epoch, ave_tower_loss, ave_tower_acc, bar_info)
