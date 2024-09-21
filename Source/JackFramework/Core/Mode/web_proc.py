# -*- coding: utf-8 -*-
from JackFramework.SysBasic.log_handler import LogHandler as log
from JackFramework.Web.web_server import WebServer

from ._meta_interface_mode import InterfaceMode


class WebProc(InterfaceMode):
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

    def exec(self, rank: int = None) -> None:
        assert rank is None
        self._init_data_model_handler(rank)
        log.info('web mode starts')
        self.__init_setting()
        self._web_server.exec()
        log.info('web mode has exited!')
