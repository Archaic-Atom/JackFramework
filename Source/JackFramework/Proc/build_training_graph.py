# -*- coding: utf-8 -*-
import os
import linecache
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import JackFramework.SysBasic.define as sysdefine
from JackFramework.SysBasic.loghander import LogHandler as log
from JackFramework.SysBasic.device_manager import DeviceManager


from JackFramework.FileHandler.model_saver import ModelSaver
from JackFramework.Evaluation.listhandler import ListHandler


class BuildGraph(object):
    """docstring for ClassName"""
    OPT_LOSS_ID = 0
    OPT_LR_GROUP_ID = 0
    DEFAULT_RANK_ID = 0

    def __init__(self, args: object, jf_model: object, rank: object) -> object:
        super().__init__()
        self.__args = args
        self.__rank = rank
        self.__jf_model = jf_model
        self.__device_manager = DeviceManager(args)
        if args.dist:
            self.__device_manager.init_distributed_gpu_device(rank)
        self.__device = self.__device_manager.device

        self.__model = None
        self.__opt = None
        self.__sch = None
        self.__init_training_graph()

    def __init_training_graph(self) -> None:
        self.__model = self.__init_model()
        self.__pass_model2device()
        self.__opt, self.__sch = self.__init_optimizer()
        self.count_parameter_num()

    def __pass_model2device(self) -> None:
        log.info("Loading model to GPUs!")
        # if self.__args.gpu > 1:
        args = self.__args

        if args.dist:
            for i, model_item in enumerate(self.__model):
                model_item = model_item.to(self.__rank)
                self.__model[i] = DDP(model_item,  device_ids=[self.__rank],
                                      find_unused_parameters=True)
        else:
            for i, model_item in enumerate(self.__model):
                self.__model[i] = nn.DataParallel(model_item)

            for i, model_item in enumerate(self.__model):
                self.__model[i].to(self.__device)

        log.info("Successfully loaded the model into GPUs!")

    def __init_model(self)->object:
        log.info("Loading user's model!")
        model = self.__jf_model.get_model()
        log.info("Successfully get user's model!")
        return model

    def __init_optimizer(self)->object:
        log.info("Loading user's optimizer!")
        args = self.__args
        opt, sch = self.__jf_model.optimizer(self.__model, args.lr)
        log.info("Successfully get user's optimizer!")
        return opt, sch

    def __reduce_tensor(self, data: torch.tensor) -> torch.tensor:
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        return data

    def cleanup(self):
        self.__device_manager.cleanup()

    def restore_model(self, rank: object) -> None:
        args = self.__args
        checkpoint_path = ModelSaver.get_check_point_path(args.modelDir)

        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            for i, _ in enumerate(self.__model):
                ModelSaver.load_model(self.__model[i], checkpoint_path, rank)
                ModelSaver.load_opt(self.__opt[i], checkpoint_path, rank)
        else:
            log.warning("no checkpoint found at '{}'".format(checkpoint_path))

    def count_parameter_num(self)->None:
        for i, model_item in enumerate(self.__model):
            num_params = 0
            for param in model_item.parameters():
                num_params += param.numel()
            log.info('Model ' + str(i) + ': The total parameter - %d' % num_params)

    def save_model(self, epoch: int) -> None:
        args = self.__args
        assert len(self.__model) == len(self.__opt)
        for i, model_item in enumerate(self.__model):
            file_name = sysdefine.CHECK_POINT_NAME % (i, epoch)
            ModelSaver.save(args.modelDir, file_name,
                            {'epoch': epoch,
                             'state_dict': model_item.state_dict(),
                             'optimizer': self.__opt[i].state_dict(),
                             })

    def set_model_mode(self, is_training: bool = True) -> None:
        if self.__model is None:
            log.error("There is no mdoel!")
        if is_training:
            for i, _ in enumerate(self.__model):
                self.__model[i].train()
        else:
            for i, _ in enumerate(self.__model):
                self.__model[i].eval()

    def train_proc(self, input_data: list,
                   label_data: list,
                   is_training: bool = True):
        if is_training:
            tower_loss_iteration, tower_acc_iteration = self.train_model(input_data, label_data)
        else:
            tower_loss_iteration, tower_acc_iteration = self.val_model(input_data, label_data)

        return tower_loss_iteration, tower_acc_iteration

    def train_model(self, input_data: list, label_data: list) -> list:
        input_data = self.__pass_data2device(input_data)
        label_data = self.__pass_data2device(label_data)

        args = self.__args

        assert len(self.__model) == len(self.__opt)
        tower_loss_iteration = []
        tower_acc_iteration = []

        for i, model_item in enumerate(self.__model):
            self.__opt[i].zero_grad()
            # get ouput
            output_data = self.__jf_model.inference(model_item, input_data, i)
            # loss and acc
            loss = self.__jf_model.loss(output_data, label_data, i)
            acc = self.__jf_model.accuary(output_data, label_data, i)

            # grad and update
            loss[self.OPT_LOSS_ID].backward()
            self.__opt[i].step()

            # to show
            tower_loss_iteration.append(self.__variable2tensor(loss))
            tower_acc_iteration.append(self.__variable2tensor(acc))
            if args.dist:
                torch.cuda.synchronize()

        return tower_loss_iteration, tower_acc_iteration

    def val_model(self, input_data: list, label_data: list) -> list:
        input_data = self.__pass_data2device(input_data)
        label_data = self.__pass_data2device(label_data)

        args = self.__args

        tower_loss_iteration = []
        tower_acc_iteration = []
        with torch.no_grad():
            for i, model_item in enumerate(self.__model):
                output_data = self.__jf_model.inference(model_item, input_data, i)
                loss = self.__jf_model.loss(output_data, label_data, i)
                tower_loss_iteration.append(self.__variable2tensor(loss))
                acc = self.__jf_model.accuary(output_data, label_data, i)
                tower_acc_iteration.append(self.__variable2tensor(acc))

                if args.dist:
                    torch.cuda.synchronize()

        return tower_loss_iteration, tower_acc_iteration

    def __pass_data2device(self, data: list)->list:
        args = self.__args

        if args.dist:
            for i, data_item in enumerate(data):
                data[i] = data_item.cuda(non_blocking=True)
            #data[i] = data_item.to(self.__rank)
        else:
            assert self.__device is not None
            for i, data_item in enumerate(data):
                data[i] = data_item.to(self.__device)

        return data

    def __variable2tensor(self, data: list)-> None:
        res = []
        args = self.__args
        for _, data_item in enumerate(data):
            if args.dist:
                log_data = self.__reduce_tensor(
                    data_item.clone().detach_() / (args.gpu))
                res.append(log_data.item())
            else:
                res.append(data_item.item())
        return res

    def cal_tower_loss_acc(self, tower_loss: list, tower_acc: list,
                           tower_loss_iteration: list,
                           tower_acc_iteration: list,
                           total_iteration: int) ->list:
        tower_loss = ListHandler.double_list_add(tower_loss_iteration, tower_loss)
        tower_acc = ListHandler.double_list_add(tower_acc_iteration, tower_acc)

        ave_tower_loss = ListHandler.double_list_div(tower_loss, total_iteration)
        ave_tower_acc = ListHandler.double_list_div(tower_acc, total_iteration)
        return tower_loss, tower_acc, ave_tower_loss, ave_tower_acc

    def adjust_lr_scheduler(self, loss: list, rank: int) -> None:
        for i, sch_item in enumerate(self.__sch):
            if sch_item is None:
                return
            self.__jf_model.lr_scheduler(sch_item, float(loss[i][0]), i)
            if rank == BuildGraph.DEFAULT_RANK_ID or rank == None:
                log.info("Model " + str(i) + " Current lr: " +
                         str(self.__opt[i].param_groups[self.OPT_LR_GROUP_ID]['lr']))

    def test_model(self, input_data: list) -> list:
        input_data = self.__pass_data2device(input_data)
        outputs_data = []
        with torch.no_grad():
            for i, model_item in enumerate(self.__model):
                output_data = self.__jf_model.inference(model_item, input_data, i)
                outputs_data.append(output_data)

        return outputs_data
