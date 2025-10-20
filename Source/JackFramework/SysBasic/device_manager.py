# -*- coding: utf-8 -*-
"""GPU and distributed device utilities."""

import os
import re
import threading
import atexit
import socket
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from JackFramework.SysBasic.log_handler import LogHandler as log
from JackFramework.SysBasic._show_manager import ShowManager


class DeviceManager(object):
    """Centralise device initialisation and distributed helpers."""

    DEFAULT_OUTPUT_DEVICE = 'cuda:0'
    DEFAULT_CPU = 'cpu'
    __DEVICE_MANAGER = None
    __ATEEXIT_REGISTERED = False

    def __init__(self, args: object) -> None:
        super().__init__()
        self.__args = args
        self.__world_size = int(os.environ.get('WORLD_SIZE', max(args.gpu, 1)))
        self.__device = None if args.dist else self.__init_device()

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__DEVICE_MANAGER is None:
            cls.__DEVICE_MANAGER = object.__new__(cls)
        return cls.__DEVICE_MANAGER

    @property
    def device(self) -> Optional[torch.device]:
        return self.__device

    @staticmethod
    def __init_cudnn(is_gpu: bool) -> None:
        if is_gpu:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    def __init_device(self) -> torch.device:
        log.info('Start initializing device!')
        args = self.__args
        self.__init_cudnn(args.gpu > 0)
        device = torch.device(
            self.DEFAULT_OUTPUT_DEVICE if args.gpu > 0 else self.DEFAULT_CPU
        )
        log.info('Finish initializing device!')
        return device

    def init_distributed_gpu_device(self, rank: int) -> None:
        """Initialise CUDA + NCCL for a specific rank."""

        log.info('Start initializing distributed device!')
        if rank is None or self.__args.gpu <= 0:
            raise ValueError('Distributed initialisation requires a valid rank and GPU count > 0.')

        # Respect torchrun/elastic env if present; only provide defaults otherwise
        env = os.environ
        env.setdefault('MASTER_ADDR', str(self.__args.ip))
        env.setdefault('MASTER_PORT', str(self.__args.port))
        env.setdefault('WORLD_SIZE', str(self.__world_size))
        if env.get('RANK') is None:
            env['RANK'] = str(rank)
        if env.get('LOCAL_RANK') is None:
            env['LOCAL_RANK'] = str(rank)
        self.__init_cudnn(True)
        dist.init_process_group('nccl', rank=rank, world_size=int(env.get('WORLD_SIZE', self.__world_size)))
        try:
            local_rank_env = os.environ.get('LOCAL_RANK')
            local_rank = int(local_rank_env) if local_rank_env is not None else rank
        except ValueError:
            local_rank = rank
        torch.cuda.set_device(local_rank)
        log.info(f'Initialised distributed (backend=nccl) rank={rank}, world_size={self.__world_size}, local_rank={local_rank}')

        # Best-effort: ensure PG is destroyed on any interpreter exit path
        if not DeviceManager.__ATEEXIT_REGISTERED:
            def _dist_cleanup_on_exit() -> None:
                try:
                    if dist.is_initialized():
                        try:
                            dist.destroy_process_group()
                        except Exception:
                            pass
                except Exception:
                    pass

            atexit.register(_dist_cleanup_on_exit)
            DeviceManager.__ATEEXIT_REGISTERED = True

    def cleanup(self) -> None:
        if self.__args.dist and dist.is_initialized():
            rank = ShowManager.get_rank()
            # Optional alignment for safety; upstream callers also align.
            try:
                if hasattr(dist, 'monitored_barrier'):
                    from datetime import timedelta
                    dist.monitored_barrier(timeout=timedelta(seconds=60))
                else:
                    dist.barrier()
            except Exception:
                pass
            log.info(f'Destroying process group (rank={rank})')
            try:
                dist.destroy_process_group()
                log.info(f'Destroyed process group (rank={rank})')
            except Exception as exc:
                log.warning(f'Process group shutdown failed on rank={rank}: {exc}')

    @staticmethod
    def check_cuda(args: object) -> bool:
        if args.gpu == 0:
            log.info('We will use cpu!')
            return True

        if not torch.cuda.is_available():
            log.error("Torch is reporting that CUDA isn't available")
            return False

        log.info(f"We detect the gpu device: {torch.cuda.get_device_name(0)}")
        log.info(f"We detect the number of gpu device: {torch.cuda.device_count()}")
        args, res_bool = DeviceManager.check_cuda_count(args)
        return res_bool

    @staticmethod
    def check_cuda_count(args: object) -> Tuple[object, bool]:
        res_bool = True
        device_count = torch.cuda.device_count()
        if device_count < args.gpu:
            log.warning('Requested GPUs exceed availability: '
                        f'{args.gpu} vs {device_count}. Falling back to available devices.')
            args.gpu = device_count

            if args.dist:
                args.port, res_bool = DeviceManager.find_unused_port(args.port)

        return args, res_bool

    @staticmethod
    def check_port_in_use(port: str, host: str = '127.0.0.1') -> bool:
        try:
            with socket.create_connection((host, int(port)), timeout=1):
                return True
        except (OSError, ValueError):
            return False

    @staticmethod
    def find_unused_port(port: str) -> Tuple[str, bool]:
        max_attempts = 5
        current_port = int(port)

        for _ in range(max_attempts):
            if not DeviceManager.check_port_in_use(str(current_port)):
                log.info(f'We will use the port: {current_port}')
                return str(current_port), True

            log.warning(f'Port: {current_port} is in use; probing the next port.')
            current_port += 1

        log.error('Unable to find an unused port within the retry limit.')
        return str(current_port), False
