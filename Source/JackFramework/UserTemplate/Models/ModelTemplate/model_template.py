# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ModelHandlerTemplate(object):
    __metaclass__ = ABCMeta

    def __init__(self, args: object) -> object:
        super().__init__()
        self.__args = args

    @abstractmethod
    def get_model(self) -> list:
        # return model's list
        pass

    @abstractmethod
    def interence(self, model: list, input_data: list, model_id: int) -> list:
        pass

    @abstractmethod
    def optimizer(self, model: list, lr: float) -> list:
        # return optimizer's list
        # return scheduler's list
        pass

    @abstractmethod
    def lr_scheduler(self, sch: object, ave_loss: float, sch_id: int) -> None:
        pass

    @abstractmethod
    def accuary(self, output_data: list, label_data: list, model_id: int) -> list:
        # return accuary's list
        pass

    @abstractmethod
    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss's list
        pass
