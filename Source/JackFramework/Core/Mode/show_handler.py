# -*- coding: utf-8 -*-
import time
from JackFramework.SysBasic.processbar import ShowProcess


class ShowHandler(object):
    def __init__(self):
        super().__init__()
        self.__process_bar = None
        self.__start_time = None
        self.__duration = None
        self.__rest_time = None

    def _init_show_setting(self, training_iteration: int, bar_info: str) -> tuple:
        self.__process_bar = ShowProcess(training_iteration, bar_info)
        self.__start_time = time.time()

    def _calculate_ave_runtime(self, total_iteration: int,
                               training_iteration: int) -> tuple:
        self.__duration = (time.time() - self.__start_time) / (total_iteration)
        self.__rest_time = (training_iteration - total_iteration) * self.__duration

    def _stop_show_setting(self) -> None:
        self.__process_bar.close()

    def duration(self):
        return time.time() - self.__start_time

    def _update_show_bar(self, info_str: str) -> None:
        self.__process_bar.show_process(show_info=info_str,
                                        rest_time=self.__rest_time,
                                        duration=self.__duration)
