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

    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = False) -> object:
        super().__init__()
        log.warning('background mode is not supperst distributed')
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

    def init_datahandler_modelhandler(self) -> tuple:
        self.__data_manager, self.__graph = self.__init_datahandler_modelhandler()

    def __get_graph_and_data_manager(self):
        return self.__graph, self.__data_manager

    def __init_setting(self, rank: object) -> tuple:
        graph, data_manager = self.__get_graph_and_data_manager()
        graph.restore_model(rank)
        graph.set_model_mode(False)
        graph.pretreatment(None, rank)

        self.__named_pipe = NamedPipe('server')
        return data_manager, self.__named_pipe

    def __testing_data_proc(self, batch_data: list) -> tuple:
        graph, data_manager = self.__get_graph_and_data_manager()
        input_data, supplement = data_manager.split_data(batch_data, False)
        outputs_data = graph.test_model(input_data)
        return outputs_data, supplement

    def __save_result(self, outputs_data: list, supplement: list, msg: str) -> None:
        _, data_manager = self.__get_graph_and_data_manager()
        data_manager.save_test_data(outputs_data, supplement, msg)
        log.info('jf server has save the results')

    def exec(self, rank: object = None) -> None:
        assert rank is None and self.__named_pipe is None
        log.info('background mode is starting')
        data_manager, named_pipe = self.__init_setting(rank)

        while(True):
            msg = named_pipe.recive()
            named_pipe.send(self.__RELY_MSG % msg)

            if msg == self.__EXIT_COMAND:
                break
            log.info('jf get message: %s' % msg)

            batch_data = data_manager.load_test_data(msg)
            outputs_data, supplement = self.__testing_data_proc(batch_data)
            self.__save_result(outputs_data, supplement, msg)
            named_pipe.send(self.__RELY_FINISH)

        log.info('background mode is exiting')
