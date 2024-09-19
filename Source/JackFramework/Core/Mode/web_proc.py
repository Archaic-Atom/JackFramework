# -*- coding: utf-8 -*-
from JackFramework.SysBasic.log_handler import LogHandler as log
from JackFramework.Web.web_server import WebServer

from .test_proc import TestProc


class WebProc(TestProc):
    """docstring for ClassName"""
    __WEB_HANDLER = None

    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = False) -> None:
        super().__init__(args, user_inference_func, is_training)
        log.warning('WebProc mode does not support distributed')
        self._web_server = WebServer(args)

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__WEB_HANDLER is None:
            cls.__WEB_HANDLER = object.__new__(cls)
        return cls.__WEB_HANDLER

    @staticmethod
    def get_web_object() -> object:
        return WebProc.__WEB_HANDLER

    def __init_setting(self) -> object:
        graph = self._graph
        graph.restore_model()
        graph.set_model_mode(False)
        graph.user_pretreatment(None)

    def __save_result(self, outputs_data: list, supplement: list, msg: str) -> None:
        _, data_manager = self._get_graph_and_data_manager
        res = data_manager.user_save_test_data(outputs_data, supplement, msg)
        log.info('jf server has saved the results')
        return res

    def __try_load_data(self, msg: str) -> tuple:
        res = True
        try:
            _, data_manager = self._get_graph_and_data_manager
            batch_data = data_manager.user_load_test_data(msg)
        except Exception:
            log.error('Any error of load_test_data function or split in dataloader!')
            res = False
            batch_data = None
        return res, batch_data

    def __try_exec_testing_proc(self, batch_data: list) -> tuple:
        res, outputs_data, supplement = True, None, None

        try:
            outputs_data, supplement = self._testing_data_proc(batch_data)
        except Exception:
            log.error('Any error of inference function in model!')
            res = False
        return res, outputs_data, supplement

    def __try_save_result(self, msg: str, outputs_data: list, supplement: list) -> bool:
        res = True
        res = self.__save_result(outputs_data, supplement, msg)
        try:
            res = self.__save_result(outputs_data, supplement, msg)
        except Exception:
            log.error('Any error of save_test_data function in dataloader!')
            res = False
        return res

    def data_handler(self, msg: str) -> bool:
        res, batch_data = self.__try_load_data(msg)
        if not res:
            return res
        res, outputs_data, supplement = self.__try_exec_testing_proc(batch_data)
        if not res:
            return res
        res = self.__try_save_result(msg, outputs_data, supplement)
        return res

    def exec(self, rank: int = None) -> None:
        assert rank is None
        self._init_data_model_handler(rank)
        log.info('web mode starts')
        self.__init_setting()
        self._web_server.exec()
        log.info('web mode has exited!')
