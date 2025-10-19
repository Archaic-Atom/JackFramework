# -*- coding: UTF-8 -*-
"""Entry point orchestration for JackFramework applications."""

import os
from collections.abc import Callable
from typing import Optional

import torch.multiprocessing as mp

from JackFramework.SysBasic.args_parser import ArgsParser
from JackFramework.SysBasic.init_handler import InitProgram
from JackFramework.SysBasic.log_handler import LogHandler as log

from .Mode import mode_selection


class Application(object):
    """Singleton wrapper coordinating argument parsing and mode execution."""

    __APPLICATION: Optional['Application'] = None

    def __new__(cls, *args: object, **kwargs: object) -> 'Application':
        if cls.__APPLICATION is None:
            cls.__APPLICATION = super().__new__(cls)
        return cls.__APPLICATION

    def __init__(self, user_interface: object, application_name: str = "") -> None:
        super().__init__()
        self.__user_interface = user_interface
        self.__application_name = application_name

    def set_user_interface(self, user_interface: object) -> None:
        self.__user_interface = user_interface

    def start(self) -> None:
        if self.__user_interface is None:
            raise RuntimeError('User interface has not been configured for the application.')

        args = self.__parse_args()
        if not InitProgram(args).init_program():
            log.error('Initialisation failed; aborting application start.')
            return

        mode_func = mode_selection(args, self.__user_interface.inference, args.mode)
        args.world_size = max(args.gpu, 1) * max(getattr(args, 'nodes', 1), 1)
        self._dist_app_start(mode_func, args)
        log.info('The Application has finished successfully.')

    def __parse_args(self):
        parser_callback = getattr(self.__user_interface, 'user_parser', None)
        return ArgsParser().parse_args(self.__application_name, parser_callback)

    @staticmethod
    def _dist_app_start(mode_func: Callable, args: object) -> None:
        if not getattr(args, 'dist', False):
            mode_func()
            return

        gpu_num = getattr(args, 'gpu', 0)
        if gpu_num <= 0:
            raise ValueError('Distributed mode requires a positive `gpu` argument.')

        nodes = max(getattr(args, 'nodes', 1), 1)
        node_rank = max(getattr(args, 'node_rank', 0), 0)
        world_size = getattr(args, 'world_size', gpu_num * nodes)

        os.environ.setdefault('MASTER_ADDR', str(args.ip))
        os.environ.setdefault('MASTER_PORT', str(args.port))
        os.environ.setdefault('WORLD_SIZE', str(world_size))

        local_rank_env = os.environ.get('LOCAL_RANK')
        rank_env = os.environ.get('RANK')
        if local_rank_env is not None:
            local_rank = int(local_rank_env)
            rank = int(rank_env) if rank_env is not None else local_rank
            mode_func(rank)
            return

        def _wrapped(local_rank: int) -> None:
            global_rank = node_rank * gpu_num + local_rank
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['RANK'] = str(global_rank)
            mode_func(global_rank)

        mp.spawn(_wrapped, nprocs=gpu_num, join=True)
