# -*- coding: utf-8 -*-
import time
from functools import wraps

from JackFramework.SysBasic.loghander import LogHandler as log
from JackFramework.FileHandler.tensorboard_handler import TensorboardHandler
from .processbar import ShowProcess


class ShowHandler(object):
    __ShowHandler = None
    __DEFAULT_RANK_ID = 0
    __RANK = None
    __PROCESS_BAR, __START_TIME = None, None
    __DURATION, __REST_TIME = None, None
    __TENSORBOARD_HANDLER = None

    def __init__(self) -> object:
        super().__init__()

    @property
    def rank(self) -> object:
        return self.__RANK

    @staticmethod
    def get_rank() -> object:
        return ShowHandler.__RANK

    @property
    def default_rank_id(self):
        return self.__DEFAULT_RANK_ID

    @staticmethod
    def set_rank(rank: object) -> None:
        ShowHandler.__RANK = rank

    @staticmethod
    def set_default_rank_id(default_rank_id: int) -> None:
        ShowHandler.__DEFAULT_RANK_ID = default_rank_id

    @staticmethod
    def init_show_setting(training_iteration: int, bar_info: str) -> tuple:
        ShowHandler.__PROCESS_BAR = ShowProcess(training_iteration, bar_info)
        ShowHandler.__START_TIME = time.time()

    @staticmethod
    def init_tensorboard_handler(args: object) -> None:
        ShowHandler.__TENSORBOARD_HANDLER = TensorboardHandler(args)

    @staticmethod
    def calculate_ave_runtime(total_iteration: int,
                              training_iteration: int) -> tuple:
        ShowHandler.__DURATION = (time.time() - ShowHandler.__START_TIME) / (total_iteration)
        ShowHandler.__REST_TIME = (training_iteration - total_iteration) * ShowHandler.__DURATION

    @staticmethod
    def stop_show_setting() -> None:
        ShowHandler.__PROCESS_BAR.close()

    @staticmethod
    def duration():
        return time.time() - ShowHandler.__START_TIME

    @classmethod
    def show_method(cls, func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if cls.__RANK == cls.__DEFAULT_RANK_ID or cls.__RANK is None:
                func(*args, **kwargs)
        return wrapped_func

    @staticmethod
    def update_show_bar(info_str: str) -> None:
        ShowHandler.__PROCESS_BAR.show_process(show_info=info_str,
                                               rest_time=ShowHandler.__REST_TIME,
                                               duration=ShowHandler.__DURATION)

    @staticmethod
    def write_tensorboard(epoch: int, ave_tower_loss: list,
                          ave_tower_acc: list, bar_info: str) -> None:
        ShowHandler.__TENSORBOARD_HANDLER.write_data(
            epoch, ave_tower_loss, ave_tower_acc, bar_info)

    def reinit_log_tensorboard_handler(self, args: object) -> None:
        if not args.dist:
            self.init_tensorboard_handler(args)
        elif self.rank == self.default_rank_id:
            # dist reinit log
            log().init_log(args.outputDir, args.pretrain)
            log().info("LogHandler is reinitialized!")
            self.init_tensorboard_handler(args)
