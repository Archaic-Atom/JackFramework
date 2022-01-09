# -*- coding: utf-8 -*-
import time
from .build_training_graph import BuildGraph
from .data_handler_manager import DataHandlerManager

from JackFramework.SysBasic.loghander import LogHandler as log
from JackFramework.Tools.process_comm import NamedPipe


class BackGround(object):
    __EXIT_COMAND = 'jf stop'
    __RELY_MSG = 'the server has recived message: %s'
    __RELY_FINISH = 'jf finish'
    __RELY_EEROR = 'jf error'

    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = False) -> object:
        super().__init__()
        log.warning('background mode does not support distributed')
        assert not args.dist and not is_training and args.batchSize == 1
        self.__args = args
        self.__is_training = is_training
        self.__user_inference_func = user_inference_func
        self.__data_manager = None
        self.__graph = None
        self.init_datahandler_modelhandler()
        self.__named_pipe = None

    def __init_datahandler_modelhandler(self) -> object:
        model, dataloader = self.__user_inference_func(self.__args)
        assert model is not None and dataloader is not None

        graph = BuildGraph(self.__args, model, None)
        data_manager = DataHandlerManager(self.__args, dataloader)
        return data_manager, graph

    def __get_graph_and_data_manager(self):
        return self.__graph, self.__data_manager

    def __init_setting(self, rank: object) -> object:
        graph, _ = self.__get_graph_and_data_manager()
        graph.restore_model(rank)
        graph.set_model_mode(False)
        graph.pretreatment(None, rank)

        self.__named_pipe = NamedPipe('server')
        return self.__named_pipe

    def __testing_data_proc(self, batch_data: list) -> tuple:
        graph, data_manager = self.__get_graph_and_data_manager()
        input_data, supplement = data_manager.split_data(batch_data, False)
        outputs_data = graph.test_model(input_data)
        return outputs_data, supplement

    def __save_result(self, outputs_data: list, supplement: list, msg: str) -> None:
        _, data_manager = self.__get_graph_and_data_manager()
        data_manager.save_test_data(outputs_data, supplement, msg)
        log.info('jf server has saved the results')

    def __try_load_data(self, msg: str) -> tuple:
        res = True
        try:
            _, data_manager = self.__get_graph_and_data_manager()
            batch_data = data_manager.load_test_data(msg)
        except Exception:
            log.error('Any error of load_test_data funcution or split in dataloader!')
            res = False
        return res, batch_data

    def __try_exec_testing_proc(self, batch_data: tuple) -> tuple:
        res, outputs_data, supplement = True, None, None
        try:
            outputs_data, supplement = self.__testing_data_proc(batch_data)
        except Exception:
            log.error('Any error of inference funcution in model!')
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
        if msg == self.__EXIT_COMAND:
            log.info('background mode is exiting!')
            return True
        return False

    def __info_processing_loop(self, named_pipe: object) -> None:
        while(True):
            msg = self.__msg_handler(named_pipe)

            res = self.__exit_cmd(msg)
            if res:
                break

            res = self.__data_handler(msg)

            if res:
                named_pipe.send(self.__RELY_FINISH)
            else:
                named_pipe.send(self.__RELY_EEROR)

    def init_datahandler_modelhandler(self) -> tuple:
        self.__data_manager, self.__graph = self.__init_datahandler_modelhandler()

    def exec(self, rank: object = None) -> None:
        assert rank is None and self.__named_pipe is None
        log.info('background mode starts')
        named_pipe = self.__init_setting(rank)

        self.__info_processing_loop(named_pipe)
        log.info('background mode has exited!')
