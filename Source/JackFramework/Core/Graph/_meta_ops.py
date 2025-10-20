# -*- coding: utf-8 -*-
import os
from abc import ABCMeta, abstractmethod
from typing import TypeVar

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import JackFramework.SysBasic.define as sys_define
from JackFramework.SysBasic.log_handler import LogHandler as log
from JackFramework.SysBasic.device_manager import DeviceManager
from JackFramework.SysBasic.show_handler import ShowHandler
from JackFramework.FileHandler.model_saver import ModelSaver

from ._user_model import UserModel

ModelHandlerTemplate = TypeVar('ModelHandlerTemplate')


class MetaOps(UserModel):
    __metaclass__ = ABCMeta
    __OPT_LR_GROUP_ID = 0

    def __init__(self, args: object, jf_model: ModelHandlerTemplate) -> None:
        super().__init__(args, jf_model)
        self.__args = args
        self.__device_manager = DeviceManager(args)
        if args.dist:
            self.__device_manager.init_distributed_gpu_device(self.rank)
        self.__device = self.__device_manager.device
        self.__init_training_graph()

    def __init_training_graph(self) -> None:
        self.user_init_model()
        self._pass_model2device()
        self.user_init_optimizer()
        self.count_parameter_num()

    def _init_ddp_model(self) -> None:
        if self._model is None:
            raise RuntimeError('Model must be initialised before wrapping with DDP.')
        if self.rank is None:
            raise ValueError('Distributed training requires a valid rank assignment.')
        for i, model_item in enumerate(self._model):
            model_item = model_item.to(self.rank)
            self._model[i] = DDP(model_item, device_ids=[self.rank])
            # self._model[i] = DDP(model_item, device_ids=[self.rank],
            #                     find_unused_parameters=self.__args.debug)
            # pytorch 2.4 can not use find_unused_parameters.

    def _init_dp_model(self) -> None:
        if self._model is None:
            raise RuntimeError('Model must be initialised before wrapping or moving to device.')

        use_cuda = (self.__args.gpu > 0) and torch.cuda.is_available()
        # If multiple GPUs requested and CUDA available, wrap with DataParallel.
        # For single GPU or CPU, keep the raw module and move to the resolved device.
        if use_cuda and max(self.__args.gpu, 0) > 1:
            for i, model_item in enumerate(self._model):
                self._model[i] = nn.DataParallel(model_item)
            for i, _ in enumerate(self._model):
                self._model[i].to(self.__device)
        else:
            for i, model_item in enumerate(self._model):
                self._model[i] = model_item.to(self.__device)

    def _pass_model2device(self) -> None:
        log.info("Loading model to GPUs!")
        if self.__args.dist:
            self._init_ddp_model()
        else:
            self._init_dp_model()
        log.info("Successfully loaded the model into GPUs!")

    def _pass_data2device(self, data: list) -> list:
        if self.__args.dist:
            for i, data_item in enumerate(data):
                if data_item is None:
                    continue
                data[i] = data_item.cuda(non_blocking=True)
        else:
            if self.__device is None:
                raise RuntimeError('Device manager did not provide a valid device.')
            for i, data_item in enumerate(data):
                if data_item is None:
                    continue
                data[i] = data_item.to(self.__device)
        return data

    def _variable2tensor(self, data: list) -> list:
        res = []
        for data_item in data:
            if self.__args.dist:
                if not torch.is_tensor(data_item):
                    raise TypeError('Distributed aggregation expects tensor inputs.')
                tensor = data_item.clone().detach()
                if not tensor.is_cuda:
                    if self.rank is None:
                        raise RuntimeError('Distributed aggregation without rank assignment.')
                    tensor = tensor.to(torch.device('cuda', self.rank))
                reduced = self._reduce_tensor(tensor)
                world_size = max(int(os.environ.get('WORLD_SIZE', self.__args.gpu)), 1)
                res.append((reduced / world_size).item())
            else:
                res.append(data_item.item())
        return res

    def _restore_model_opt(self, checkpoint: dict) -> None:
        for i, _ in enumerate(self._model):
            if not self.user_load_model(checkpoint, i):
                ModelSaver.load_model(self._model[i], checkpoint, i)
            if not self.user_load_opt(checkpoint, i):
                ModelSaver.load_opt(self._opt[i], checkpoint, i)

    @ShowHandler.show_method
    def show_lr_scheduler_info(self, idx: int) -> None:
        log.info((f'Model_{idx} Current lr: ' +
                  str(self._opt[idx].param_groups[self.__OPT_LR_GROUP_ID]['lr'])))

    @ShowHandler.show_method
    def count_parameter_num(self) -> None:
        for i, model_item in enumerate(self._model):
            num_params = sum(param.numel() for param in model_item.parameters())
            log.info(f'Model {str(i)}' + f': The total parameter - {num_params}')

    def adjust_lr_scheduler(self, loss: list) -> None:
        for i, sch_item in enumerate(self._sch):
            if sch_item is not None:
                self.user_lr_scheduler(sch_item, loss, i)
                self.show_lr_scheduler_info(i)

    def cleanup(self):
        # Try to complete outstanding CUDA work before teardown
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        # Release model/optimizers so DDP wrappers can be GC'ed prior to PG destroy
        try:
            self.free_model()
        except Exception:
            pass
        # Finally, destroy the process group/backend
        self.__device_manager.cleanup()

    def restore_model(self) -> None:
        checkpoint_path = ModelSaver.get_check_point_path(self.__args.modelDir)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            checkpoint = ModelSaver.load_checkpoint(checkpoint_path, self.rank)
            self._restore_model_opt(checkpoint)
        else:
            log.warning('No checkpoint found; starting from scratch.')

    def save_model(self, epoch: int) -> None:
        if len(self._model) != len(self._opt):
            raise ValueError('Model and optimizer collections must have the same length.')
        file_name = sys_define.CHECK_POINT_NAME % epoch
        model_dict = self.user_save_model(epoch)
        if model_dict is None:
            model_dict = ModelSaver.construct_model_dict(epoch, self._model, self._opt)
        ModelSaver.save(self.__args.modelDir, file_name, model_dict)

    def set_model_mode(self, is_training: bool = True) -> None:
        if self._model is None:
            raise RuntimeError('Model has not been initialised.')
        for i, _ in enumerate(self._model):
            if is_training:
                self._model[i].train()
            else:
                self._model[i].eval()

    @staticmethod
    def _reduce_tensor(data: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        return data

    @abstractmethod
    def exec(self, input_data: list, label_data: list, is_training: bool = True) -> list:
        pass
