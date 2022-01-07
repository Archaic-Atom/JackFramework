# -*- coding: utf-8 -*-
import os
import socket
import torch
import torch.distributed as dist
from JackFramework.SysBasic.loghander import LogHandler as log


class DeviceManager(object):
    """docstring for DeviceManager"""
    DEFAULT_OUTPUT_DEVICE = 'cuda:0'
    __DEVICE_MANAGER = None

    def __init__(self, args: object):
        super().__init__()
        self.__args = args

        self.__device = self.__init_gpu_device() if not args.dist else None

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__DEVICE_MANAGER is None:
            cls.__DEVICE_MANAGER = object.__new__(cls)
        return cls.__DEVICE_MANAGER

    @property
    def device(self):
        return self.__device

    def __init_gpu_device(self) -> object:
        log.info("Start initializing device!")

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        device = torch.device(DeviceManager.DEFAULT_OUTPUT_DEVICE)

        log.info("Finish initializing device!")
        return device

    def init_distributed_gpu_device(self, rank: int) -> None:
        log.info("Start initializing distributed device!")
        assert rank is not None

        args = self.__args
        os.environ['MASTER_ADDR'] = args.ip
        os.environ['MASTER_PORT'] = args.port

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        dist.init_process_group("nccl", rank=rank, world_size=self.__args.gpu)
        torch.cuda.set_device(rank)

        return None

    def cleanup(self):
        args = self.__args
        if args.dist:
            dist.destroy_process_group()

    @staticmethod
    def check_cuda(args):
        if not torch.cuda.is_available():
            log.error("Torch is reporting that CUDA isn't available")
            return False
        log.info("We detect the gpu device: " + torch.cuda.get_device_name(0))
        log.info("We detect the number of gpu device: " + str(torch.cuda.device_count()))
        args, res_bool = DeviceManager.check_cuda_count(args)
        return res_bool

    @staticmethod
    def check_cuda_count(args) -> object:
        res_bool = True

        if torch.cuda.device_count() < args.gpu:
            log.warning(
                "The setting of GPUs is more than actually owned GPUs: %d vs %d"
                % (args.gpu, torch.cuda.device_count()))
            log.info("We will use all actually owned GPUs.")
            args.gpu = torch.cuda.device_count()

            if args.dist:
                args.port, res_bool = DeviceManager.find_unused_port(args.port)

        return args, res_bool

    @staticmethod
    def check_port_in_use(port: str, host: str = '127.0.0.1') -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((host, int(port)))
            s.settimeout(1)
            s.shutdown(2)
            return True
        except ValueError:
            return False

    @staticmethod
    def find_unused_port(port: str) -> bool:
        max_failed_num = 5
        try_index = 0
        off_set = 1
        find_res_bool = False
        while True:
            try_index += off_set
            res_bool = True
            if DeviceManager.check_port_in_use(port):
                log.warning("Port: " + str(port) + " is using")
                port = str(int(port) + off_set)
                res_bool = False

            if res_bool:
                log.info("We will use the port: " + str(port))
                find_res_bool = True
                break

            if try_index >= max_failed_num:
                log.error("We do not find unused port!")
                break

        return port, find_res_bool
