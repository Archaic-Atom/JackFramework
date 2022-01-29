# -*- coding: utf-8 -*-
import torch

from ._meta_ops import MetaOps
from ._loss_computer import ResultContainer


class BuildTrainingGraph(MetaOps, ResultContainer):
    """docstring for ClassName"""
    OPT_LOSS_ID = 0

    def __init__(self, args: object, jf_model: object) -> object:
        MetaOps.__init__(self, args, jf_model)
        ResultContainer.__init__(self)
        self.__args = args

    def __calculation_process(self, model_item: object, input_data: list, label_data: list,
                              model_id: int, is_training: bool = True) -> tuple:
        output_data, loss, acc = self.__init_calculation_result()
        output_data = self.user_inference(model_item, input_data, model_id)
        if is_training:
            loss = self.user_loss(output_data, label_data, model_id)
            acc = self.user_accuary(output_data, label_data, model_id)
        return output_data, loss, acc

    def __pass_input_label2device(self, input_data: torch.tensor,
                                  label_data: torch.tensor) -> tuple:
        input_data = self._pass_data2device(input_data)
        label_data = self._pass_data2device(label_data)
        return input_data, label_data

    def __update_model(self, loss: list, model_id: int) -> None:
        loss[self.OPT_LOSS_ID].backward()
        self._opt[model_id].step()

    def __synchronize_data(self) -> None:
        if self.__args.dist:
            torch.cuda.synchronize()

    def _train_model(self, input_data: list, label_data: list) -> list:
        assert len(self._model) == len(self._opt)
        self.init_tower_loss_and_tower_acc()
        input_data, label_data = self.__pass_input_label2device(input_data, label_data)

        for i, model_item in enumerate(self._model):
            self._opt[i].zero_grad()
            _, loss, acc = self.__calculation_process(model_item, input_data, label_data, i)
            self.__update_model(loss, i)
            self.append_iteration_loss_acc(self._variable2tensor(loss), self._variable2tensor(acc))
            self.__synchronize_data()

    def _val_model(self, input_data: list, label_data: list) -> list:
        self.init_tower_loss_and_tower_acc()
        input_data, label_data = self.__pass_input_label2device(input_data, label_data)
        with torch.no_grad():
            for i, model_item in enumerate(self._model):
                _, loss, acc = self.__calculation_process(model_item, input_data, label_data, i)
                self.append_iteration_loss_acc(self._variable2tensor(loss),
                                               self._variable2tensor(acc))
                self.__synchronize_data()

    def exec(self, input_data: list, label_data: list, is_training: bool = True):
        if is_training:
            self._train_model(input_data, label_data)
        else:
            self._val_model(input_data, label_data)

    @staticmethod
    def __init_calculation_result():
        output_data, loss, acc = None, None, None
        return output_data, loss, acc
