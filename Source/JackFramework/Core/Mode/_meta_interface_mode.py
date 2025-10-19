# -*- coding: utf-8 -*-
from collections.abc import Callable
from typing import Any

from JackFramework.SysBasic.log_handler import LogHandler as log

from .test_proc import TestProc


def error_handler(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error(f"Error in {func.__name__}: {exc}")
            return False

    return wrapper


class InterfaceMode(TestProc):
    ID_OUTPUTS_DATA, ID_SUPPLEMENT = 0, 1

    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = False) -> None:
        super().__init__(args, user_inference_func, is_training)
        log.info('Create InterfaceMode')

    def __save_result(self, outputs_data: list, supplement: list, msg: str):
        res = self._data_manager.user_save_test_data(outputs_data, supplement, msg)
        log.info('The server has saved the inference results.')
        return res

    @error_handler
    def __try_load_data(self, msg: str) -> tuple:
        return self._data_manager.user_load_test_data(msg)

    @error_handler
    def __try_exec_testing_proc(self, batch_data: list) -> tuple:
        return self._testing_data_proc(batch_data)

    @error_handler
    def __try_save_result(self, msg: str, outputs_data: list, supplement: list) -> bool:
        return self.__save_result(outputs_data, supplement, msg)

    def data_handler(self, msg: str) -> bool:
        batch_data = self.__try_load_data(msg)
        if batch_data is False:
            return False

        res = self.__try_exec_testing_proc(batch_data)
        if res is False:
            return False

        return self.__try_save_result(msg, res[self.ID_OUTPUTS_DATA], res[self.ID_SUPPLEMENT])
