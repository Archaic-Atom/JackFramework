# -*- coding: utf-8 -*-
from JackFramework.SysBasic.log_handler import LogHandler as log
from JackFramework.Tools.process_comm import NamedPipe

from ._meta_interface_mode import InterfaceMode


class BackGround(InterfaceMode):
    __EXIT_COMMAND = 'jf stop'
    __RELY_MSG = 'the server has got message: %s'
    __RELY_FINISH = 'jf finish'
    __RELY_ERROR = 'jf error'

    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = False) -> None:
        super().__init__(args, user_inference_func, is_training)
        log.warning('background mode does not support distributed')
        if args.dist or is_training or args.batchSize != 1:
            raise ValueError('background mode requires non-distributed inference with batchSize == 1.')
        self.__named_pipe = None

    def __init_setting(self) -> object:
        if self._graph is None:
            raise RuntimeError('Graph not initialised before starting background mode.')
        graph = self._graph
        graph.restore_model()
        graph.set_model_mode(False)
        graph.user_pretreatment(None)
        self.__named_pipe = NamedPipe('server')
        return self.__named_pipe

    def __msg_handler(self, named_pipe: object) -> str:
        msg = named_pipe.receive()
        named_pipe.send(self.__RELY_MSG % msg)
        log.info(f'jf gets message: {msg}')
        return msg

    def __exit_cmd(self, msg: str) -> bool:
        return msg == self.__EXIT_COMMAND

    def __info_processing_loop(self, named_pipe: object) -> None:
        while True:
            msg = self.__msg_handler(named_pipe)
            if (res := self.__exit_cmd(msg)):
                log.info(f'the result is {res}, background mode is exiting!')
                break
            if (res := self.data_handler(msg)):
                log.info(f'the result is {res}, the server is sending msg!')
                named_pipe.send(self.__RELY_FINISH)
            else:
                named_pipe.send(self.__RELY_ERROR)

    def exec(self, rank: int = None) -> None:
        if rank is not None:
            raise ValueError('background mode runs on a single process and does not accept rank.')
        if self.__named_pipe is not None:
            raise RuntimeError('background mode has already been initialised.')
        self._init_data_model_handler(rank)
        log.info('background mode starts')
        named_pipe = self.__init_setting()
        self.__info_processing_loop(named_pipe)
        log.info('background mode has exited!')
