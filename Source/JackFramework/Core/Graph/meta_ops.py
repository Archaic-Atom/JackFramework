# -*- coding: utf-8 -*-
import os
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import JackFramework.SysBasic.define as sysdefine
from JackFramework.SysBasic.loghander import LogHandler as log
from JackFramework.SysBasic.device_manager import DeviceManager
from JackFramework.SysBasic.show_handler import ShowHandler

from JackFramework.FileHandler.model_saver import ModelSaver
from .user_model import UserModel


class MetaOps(UserModel):
    __metaclass__ = ABCMeta
    __OPT_LR_GROUP_ID = 0

    def __init__(self, args: object, jf_model: object) -> object:
        super().__init__(args, jf_model)
        self.__args = args
        self.__device_manager = DeviceManager(args)
        if args.dist:
            self.__device_manager.init_distributed_gpu_device(self.rank)
        self.__device = self.__device_manager.device
        self.__init_training_graph()

    def __init_training_graph(self) -> None:
        self.user_init_model()
        self.__pass_model2device()
        self.user_init_optimizer()
        self.count_parameter_num()

    def _init_ddp_model(self) -> None:
        assert self._model is not None
        for i, model_item in enumerate(self._model):
            model_item = model_item.to(self.rank)
            self._model[i] = DDP(model_item, device_ids=[self.rank],
                                 find_unused_parameters=True)

    def _init_dp_model(self) -> None:
        assert self._model is not None
        for i, model_item in enumerate(self._model):
            self._model[i] = nn.DataParallel(model_item)

        for i, model_item in enumerate(self._model):
            self._model[i].to(self.__device)

    def __pass_model2device(self) -> None:
        log.info("Loading model to GPUs!")
        args = self.__args
        if args.dist:
            self._init_ddp_model()
        else:
            self._init_dp_model()
        log.info("Successfully loaded the model into GPUs!")

    def _pass_data2device(self, data: list) -> list:
        if self.__args.dist:
            for i, data_item in enumerate(data):
                data[i] = data_item.cuda(non_blocking=True)
        else:
            assert self.__device is not None
            for i, data_item in enumerate(data):
                data[i] = data_item.to(self.__device)

        return data

    def _variable2tensor(self, data: list) -> None:
        res = []
        for data_item in data:
            if self.__args.dist:
                log_data = self._reduce_tensor(
                    data_item.clone().detach_() / (self.__args.gpu))
                res.append(log_data.item())
            else:
                res.append(data_item.item())
        return res

    @ShowHandler.show_method
    def show_lr_scheduler_info(self, idx: int) -> None:
        log.info("Model_" + str(idx) + " Current lr: " +
                 str(self._opt[idx].param_groups[self.__OPT_LR_GROUP_ID]['lr']))

    def adjust_lr_scheduler(self, loss: list) -> None:
        for i, sch_item in enumerate(self._sch):
            if sch_item is not None:
                self.user_lr_scheduler(sch_item, loss, i)
                self.show_lr_scheduler_info(i)

    def count_parameter_num(self) -> None:
        for i, model_item in enumerate(self._model):
            num_params = sum(param.numel() for param in model_item.parameters())
            log.info('Model ' + str(i) + ': The total parameter - %d' % num_params)

    def cleanup(self):
        self.__device_manager.cleanup()

    def restore_model(self) -> None:
        checkpoint_path = ModelSaver.get_check_point_path(self.__args.modelDir)
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            checkpoint = ModelSaver.load_checkpoint(checkpoint_path, self.rank)
            for i, _ in enumerate(self._model):
                if not self.user_load_model(checkpoint, i):
                    ModelSaver.load_model(self._model[i], checkpoint, i)
                if not self.user_load_opt(checkpoint, i):
                    ModelSaver.load_opt(self._opt[i], checkpoint, i)
        else:
            log.warning("no checkpoint found at '{}'".format(checkpoint_path))

    def save_model(self, epoch: int) -> None:
        assert len(self._model) == len(self._opt)
        file_name = sysdefine.CHECK_POINT_NAME % epoch
        model_dict = self.user_save_model(epoch)
        if model_dict is None:
            model_dict = ModelSaver.construct_model_dict(epoch, self._model, self._opt)
        ModelSaver.save(self.__args.modelDir, file_name, model_dict)

    def set_model_mode(self, is_training: bool = True) -> None:
        if self._model is None:
            log.error("There is no mdoel!")
        for i, _ in enumerate(self._model):
            if is_training:
                self._model[i].train()
            else:
                self._model[i].eval()

    @abstractmethod
    def exec(self, input_data: list, label_data: list, is_training: bool = True) -> list:
        pass

    @staticmethod
    def _reduce_tensor(data: torch.tensor) -> torch.tensor:
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        return data
