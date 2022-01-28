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


class MetaOps(ShowHandler):
    __metaclass__ = ABCMeta
    __OPT_LR_GROUP_ID = 0

    def __init__(self, args: object, jf_model: object) -> object:
        super().__init__()
        self.__args, self.__jf_model = args, jf_model
        self.__device_manager = DeviceManager(args)
        if args.dist:
            self.__device_manager.init_distributed_gpu_device(self.rank)
        self.__device = self.__device_manager.device
        self.__model, self.__opt, self.__sch = None, None, None
        self.__init_training_graph()

    @property
    def _model(self):
        return self.__model

    @property
    def _opt(self):
        return self.__opt

    def __init_training_graph(self) -> None:
        self.__model = self._init_model()
        self.__pass_model2device()
        self.__opt, self.__sch = self._init_optimizer()
        self.count_parameter_num()

    def _init_ddp_model(self) -> None:
        assert self.__model is not None
        for i, model_item in enumerate(self.__model):
            model_item = model_item.to(self.rank)
            self.__model[i] = DDP(model_item, device_ids=[self.rank],
                                  find_unused_parameters=True)

    def _init_dp_model(self) -> None:
        assert self.__model is not None
        for i, model_item in enumerate(self.__model):
            self.__model[i] = nn.DataParallel(model_item)

        for i, model_item in enumerate(self.__model):
            self.__model[i].to(self.__device)

    def __pass_model2device(self) -> None:
        log.info("Loading model to GPUs!")
        args = self.__args
        if args.dist:
            self._init_ddp_model()
        else:
            self._init_dp_model()
        log.info("Successfully loaded the model into GPUs!")

    def _init_model(self) -> object:
        log.info("Loading user's model!")
        model = self.__jf_model.get_model()
        log.info("Successfully get user's model!")
        return model

    def _init_optimizer(self) -> object:
        log.info("Loading user's optimizer!")
        opt, sch = self.__jf_model.optimizer(self.__model, self.__args.lr)
        log.info("Successfully get user's optimizer!")
        return opt, sch

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
                 str(self.__opt[idx].param_groups[self.__OPT_LR_GROUP_ID]['lr']))

    def adjust_lr_scheduler(self, loss: list) -> None:
        for i, sch_item in enumerate(self.__sch):
            if sch_item is not None:
                self.__jf_model.lr_scheduler(sch_item, float(loss[i][0]), i)
                self.show_lr_scheduler_info(i)

    def count_parameter_num(self) -> None:
        for i, model_item in enumerate(self.__model):
            num_params = sum(param.numel() for param in model_item.parameters())
            log.info('Model ' + str(i) + ': The total parameter - %d' % num_params)

    def cleanup(self):
        self.__device_manager.cleanup()

    def restore_model(self) -> None:
        checkpoint_path = ModelSaver.get_check_point_path(self.__args.modelDir)
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            checkpoint = ModelSaver.load_checkpoint(checkpoint_path, self.rank)
            for i, _ in enumerate(self.__model):
                if self.__jf_model.load_model(self.__model[i], checkpoint, i) is False:
                    ModelSaver.load_model(self.__model[i], checkpoint, i)
                if self.__jf_model.load_opt(self.__opt[i], checkpoint, i) is False:
                    ModelSaver.load_opt(self.__opt[i], checkpoint, i)
        else:
            log.warning("no checkpoint found at '{}'".format(checkpoint_path))

    def save_model(self, epoch: int) -> None:
        assert len(self.__model) == len(self.__opt)
        file_name = sysdefine.CHECK_POINT_NAME % epoch
        model_dict = self.__jf_model.save_model(epoch, self.__model, self.__opt)
        if model_dict is None:
            model_dict = ModelSaver.construct_model_dict(epoch, self.__model, self.__opt)
        ModelSaver.save(self.__args.modelDir, file_name, model_dict)

    def set_model_mode(self, is_training: bool = True) -> None:
        if self.__model is None:
            log.error("There is no mdoel!")
        for i, _ in enumerate(self.__model):
            if is_training:
                self.__model[i].train()
            else:
                self.__model[i].eval()

    def pretreatment(self, epoch: int) -> None:
        self.__jf_model.pretreatment(epoch, self.rank)

    def postprocess(self, epoch: int, ave_tower_loss: list = None,
                    ave_tower_acc: list = None) -> None:
        self.__jf_model.postprocess(epoch, self.rank, ave_tower_loss, ave_tower_acc)

    def inference(self, model_item: object, input_data: list, model_id: int) -> list:
        return self.__jf_model.inference(model_item, input_data, model_id)

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        return self.__jf_model.loss(output_data, label_data, model_id)

    def accuary(self, output_data: list, label_data: list, model_id: int) -> list:
        return self.__jf_model.accuary(output_data, label_data, model_id)

    @abstractmethod
    def exec(self, input_data: list, label_data: list, is_training: bool = True) -> list:
        pass

    @staticmethod
    def _reduce_tensor(data: torch.tensor) -> torch.tensor:
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        return data
