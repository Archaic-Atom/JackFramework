# -*- coding: utf-8 -*-
from JackFramework.SysBasic.loghander import LogHandler as log
from JackFramework.Tools.process_comm import NamedPipe

from .test_proc import TestProc


class BackGround(TestProc):
    __EXIT_COMMAND = 'jf stop'
    __RELY_MSG = 'the server has got message: %s'
    __RELY_FINISH = 'jf finish'
    __RELY_ERROR = 'jf error'

    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = False) -> None:
        super().__init__(args, user_inference_func, is_training)
        log.warning('background mode does not support distributed')
        assert not args.dist and not is_training and args.batchSize == 1
        self.__named_pipe = None

    def __init_setting(self) -> object:
        graph = self._graph
        graph.restore_model()
        graph.set_model_mode(False)
        graph.user_pretreatment(None)
        self.__named_pipe = NamedPipe('server')
        return self.__named_pipe

    def __save_result(self, outputs_data: list, supplement: list, msg: str) -> None:
        _, data_manager = self._get_graph_and_data_manager
        data_manager.user_save_test_data(outputs_data, supplement, msg)
        log.info('jf server has saved the results')

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
            outputs_data, supplement = self.__testing_data_proc(batch_data)
        except Exception:
            log.error('Any error of inference function in model!')
            res = False
        return res, outputs_data, supplement

    def __try_save_result(self, msg: str, outputs_data: list, supplement: list) -> bool:
        res = True
        try:
            self.__save_result(outputs_data, supplement, msg)
        except Exception:
            log.error('Any error of save_test_data funcution in dataloader!')
            res = False
        return res

    def __data_handler(self, msg: str) -> bool:
        res, batch_data = self.__try_load_data(msg)
        if not res:
            return res
        res, outputs_data, supplement = self.__try_exec_testing_proc(batch_data)
        if not res:
            return res
        res = self.__try_save_result(msg, outputs_data, supplement)
        return res

    def __msg_handler(self, named_pipe: object) -> str:
        msg = named_pipe.receive()
        named_pipe.send(self.__RELY_MSG % msg)
        log.info('jf gets message: %s' % msg)
        return msg

    def __exit_cmd(self, msg: str) -> bool:
        return msg == self.__EXIT_COMMAND

    def __info_processing_loop(self, named_pipe: object) -> None:
        while True:
            msg = self.__msg_handler(named_pipe)
            if (res := self.__exit_cmd(msg)):
                log.info('the result is %s, background mode is exiting!' % res)
                break
            if (res := self.__data_handler(msg)):
                named_pipe.send(self.__RELY_FINISH)
            else:
                named_pipe.send(self.__RELY_ERROR)

    def exec(self, rank: object = None) -> None:
        assert rank is None and self.__named_pipe is None
        self._init_data_model_handler(rank)
        log.info('background mode starts')
        named_pipe = self.__init_setting()
        self.__info_processing_loop(named_pipe)
        log.info('background mode has exited!')
