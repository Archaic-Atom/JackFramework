# -*- coding: utf-8 -*-
import torch

from JackFramework.Evaluation.listhandler import ListHandler
from .meta_ops import MetaOps


class BuildTrainingGraph(MetaOps):
    """docstring for ClassName"""
    OPT_LOSS_ID = 0

    def __init__(self, args: object, jf_model: object, rank: object) -> object:
        super().__init__(args, jf_model, rank)
        self.__args = args

    def __calculation_process(self, model_item: object, input_data: list, label_data: list,
                              model_id: int, is_training: bool = True) -> tuple:
        output_data, loss, acc = self.__init_calculation_result()

        # get ouput
        output_data = self.inference(model_item, input_data, model_id)
        # loss and acc
        if is_training:
            loss = self.loss(output_data, label_data, model_id)
            acc = self.accuary(output_data, label_data, model_id)

        return output_data, loss, acc

    def exec(self, input_data: list, label_data: list, is_training: bool = True):
        if is_training:
            tower_loss_iteration, tower_acc_iteration = self._train_model(input_data, label_data)
        else:
            tower_loss_iteration, tower_acc_iteration = self._val_model(input_data, label_data)

        return tower_loss_iteration, tower_acc_iteration

    def _train_model(self, input_data: list, label_data: list) -> list:
        assert len(self._model) == len(self._opt)
        args = self.__args
        tower_loss_iteration, tower_acc_iteration = self.__init_tower_loss_and_tower_acc()

        input_data = self._pass_data2device(input_data)
        label_data = self._pass_data2device(label_data)

        for i, model_item in enumerate(self._model):
            self._opt[i].zero_grad()
            _, loss, acc = self.__calculation_process(model_item, input_data, label_data, i)

            loss[self.OPT_LOSS_ID].backward()
            self._opt[i].step()

            tower_loss_iteration.append(self._variable2tensor(loss))
            tower_acc_iteration.append(self._variable2tensor(acc))
            if args.dist:
                torch.cuda.synchronize()

        return tower_loss_iteration, tower_acc_iteration

    def _val_model(self, input_data: list, label_data: list) -> list:
        args = self.__args
        tower_loss_iteration, tower_acc_iteration = self.__init_tower_loss_and_tower_acc()

        input_data = self._pass_data2device(input_data)
        label_data = self._pass_data2device(label_data)

        with torch.no_grad():
            for i, model_item in enumerate(self._model):
                _, loss, acc = self.__calculation_process(model_item, input_data, label_data, i)
                tower_loss_iteration.append(self._variable2tensor(loss))
                tower_acc_iteration.append(self._variable2tensor(acc))

                if args.dist:
                    torch.cuda.synchronize()

        return tower_loss_iteration, tower_acc_iteration

    @staticmethod
    def cal_tower_loss_acc(tower_loss: list, tower_acc: list, tower_loss_iteration: list,
                           tower_acc_iteration: list, total_iteration: int) -> list:
        tower_loss = ListHandler.double_list_add(tower_loss_iteration, tower_loss)
        tower_acc = ListHandler.double_list_add(tower_acc_iteration, tower_acc)

        ave_tower_loss = ListHandler.double_list_div(tower_loss, total_iteration)
        ave_tower_acc = ListHandler.double_list_div(tower_acc, total_iteration)
        return tower_loss, tower_acc, ave_tower_loss, ave_tower_acc

    @staticmethod
    def __init_tower_loss_and_tower_acc():
        tower_loss_iteration, tower_acc_iteration = [], []
        return tower_loss_iteration, tower_acc_iteration

    @staticmethod
    def __init_calculation_result():
        output_data, loss, acc = None, None, None
        return output_data, loss, acc
