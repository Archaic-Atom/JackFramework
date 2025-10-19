# -*- coding: utf-8 -*-
"""GPU and distributed device utilities."""

import os
import socket
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from JackFramework.SysBasic.log_handler import LogHandler as log


class DeviceManager(object):
    """Centralise device initialisation and distributed helpers."""

    DEFAULT_OUTPUT_DEVICE = 'cuda:0'
    DEFAULT_CPU = 'cpu'
    __DEVICE_MANAGER = None

    def __init__(self, args: object) -> None:
        super().__init__()
        self.__args = args
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

        os.environ['MASTER_ADDR'] = self.__args.ip
        os.environ['MASTER_PORT'] = str(self.__args.port)
        os.environ['RANK'] = str(rank)
        os.environ.setdefault('LOCAL_RANK', str(rank))
        os.environ.setdefault('WORLD_SIZE', str(self.__args.gpu))
        self.__init_cudnn(True)
        dist.init_process_group('nccl', rank=rank, world_size=self.__args.gpu)
        torch.cuda.set_device(rank)

    def cleanup(self) -> None:
        if self.__args.dist and dist.is_initialized():
            dist.destroy_process_group()

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
