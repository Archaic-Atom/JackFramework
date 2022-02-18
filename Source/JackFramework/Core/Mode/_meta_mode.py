# -*- coding: utf-8 -*-
import math
from abc import ABCMeta, abstractmethod

from JackFramework.SysBasic.show_handler import ShowHandler
from JackFramework.Core.Graph import graph_selection, dataloader_selection


class MetaMode(ShowHandler):
    __metaclass__ = ABCMeta

    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = True) -> None:
        super().__init__()
        self.__args = args
        self.__is_training = is_training
        self.__user_inference_func = user_inference_func
        self.__graph, self.__data_manager = None, None
        self.__training_iteration, self.__val_iteration = self.__calculate_batch_size()

    @property
    def _graph(self) -> object:
        return self.__graph

    @property
    def _data_manager(self) -> object:
        return self.__data_manager

    @property
    def _is_training(self) -> object:
        return self.__is_training

    @property
    def _training_iteration(self) -> int:
        return self.__training_iteration

    @property
    def _val_iteration(self) -> int:
        return self.__val_iteration

    @property
    def _get_graph_and_data_manager(self):
        return self.__graph, self.__data_manager

    def __calculate_batch_size(self):
        args = self.__args
        if args.dist:
            training_iteration = math.ceil(
                args.imgNum * args.sampleNum / (args.batchSize * args.gpu))
            val_iteration = math.ceil(
                args.valImgNum * args.sampleNum / (args.batchSize * args.gpu))
        else:
            training_iteration = math.ceil(args.imgNum * args.sampleNum / args.batchSize)
            val_iteration = math.ceil(args.valImgNum * args.sampleNum / args.batchSize)
        return training_iteration, val_iteration

    def _init_datahandler_modelhandler(self, rank: object) -> object:
        self.set_rank(rank)
        self.reinit_log_tensorboard_handler(self.__args)
        jf_model, jf_dataloader = self.__user_inference_func(self.__args)
        assert jf_model is not None and jf_dataloader is not None
        self.__graph = graph_selection(self.__args, jf_model)
        self.__data_manager = dataloader_selection(self.__args, jf_dataloader)

    def _get_img_id(self, iteration: int) -> int:
        if self.rank is None:
            return iteration
        return self.rank + iteration * (self.__args.batchSize * self.__args.gpu)

    def _save_result(self, iteration: int, outputs_data: list, supplement: list):
        img_id = self._get_img_id(iteration)
        self._data_manager.user_save_result(outputs_data, supplement, img_id)

    @ShowHandler.show_method
    def _save_model(self, epoch: int) -> None:
        off_set = 1
        if (epoch + off_set) % self.__args.auto_save_num == 0:
            self._graph.save_model(epoch)

    @ShowHandler.show_method
    def _write_epoch_log(self, epoch: int) -> None:
        self._data_manager.user_show_training_info(
            epoch, self._graph.ave_tower_loss, self._graph.ave_tower_acc,
            self.duration(), self._is_training)

    def set_training_iteration(self, iteration: int) -> None:
        self.__training_iteration = iteration

    def set_val_iteration(self, iteration: int) -> None:
        self.__val_iteration = iteration

    @abstractmethod
    def exec(self, rank: object = None) -> None:
        # do something in this mode
        pass
