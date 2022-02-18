# -*- coding: utf-8 -*-
import time

from JackFramework.SysBasic.loghander import LogHandler as log
from JackFramework.FileHandler.tensorboard_handler import TensorboardHandler

from .processbar import ShowProcess
from ._show_manager import ShowManager


class ShowHandler(ShowManager):
    __ShowHandler = None
    __TENSORBOARD_HANDLER = None
    __PROCESS_BAR, __START_TIME = None, None
    __DURATION, __REST_TIME = None, None

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

    @staticmethod
    def init_tensorboard_handler(args: object) -> None:
        ShowHandler.__TENSORBOARD_HANDLER = TensorboardHandler(args)

    @staticmethod
    def calculate_ave_runtime(total_iteration: int, training_iteration: int) -> None:
        ShowHandler.__DURATION = (time.time() - ShowHandler.__START_TIME) / total_iteration
        ShowHandler.__REST_TIME = (training_iteration - total_iteration) * ShowHandler.__DURATION

    @staticmethod
    def stop_show_setting() -> None:
        ShowHandler.__PROCESS_BAR.close()

    @staticmethod
    def duration():
        return time.time() - ShowHandler.__START_TIME

    @staticmethod
    def __reinit_log_handler(args: object) -> None:
        if args.dist:
            log().init_log(args.outputDir, args.pretrain)
            log().info("LogHandler is reinitialized!")

    @staticmethod
    def update_show_bar(info_str: str) -> None:
        ShowHandler.__PROCESS_BAR.show_process(show_info=info_str,
                                               rest_time=ShowHandler.__REST_TIME,
                                               duration=ShowHandler.__DURATION)

    @staticmethod
    def write_tensorboard(epoch: int, ave_tower_loss: list,
                          ave_tower_acc: list, bar_info: str) -> None:
        ShowHandler.__TENSORBOARD_HANDLER.write_data(epoch, ave_tower_loss, ave_tower_acc, bar_info)
