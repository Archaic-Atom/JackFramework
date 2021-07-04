# -*- coding: utf-8 -*-
import os
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


class TensorboardHandler(object):
    """docstring for TensorboardHandler"""
    __TENSORBOARD_HANDLER = None

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__TENSORBOARD_HANDLER is None:
            cls.__TENSORBOARD_HANDLER = object.__new__(cls)
        return cls.__TENSORBOARD_HANDLER

    def __init__(self, args: object) -> None:
        super().__init__()
        self.__arg = args
        self.__writer = SummaryWriter(log_dir=args.log)

    def _write_file(self, epoch: int, mode_id: int,
                    data_title: str, data_list: list,
                    data_state: str) -> None:
        for i, data_item in enumerate(data_list):
            self.__writer.add_scalar(data_title % (mode_id, i, data_state),
                                     data_item, epoch)

    def write_data(self, epoch: int, model_loss_list: list,
                   model_acc_list: list, data_state: str) -> None:
        data_loss_title = 'model:%d/l%d/%s'
        data_acc_title = 'model:%d/acc%d/%s'
        assert len(model_loss_list) == len(model_acc_list)

        for i, loss_list_item in enumerate(model_loss_list):
            self._write_file(epoch, i, data_loss_title, loss_list_item, data_state)
            self._write_file(epoch, i, data_acc_title, model_acc_list[i], data_state)
