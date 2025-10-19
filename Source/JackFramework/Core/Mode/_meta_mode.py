# -*- coding: utf-8 -*-
"""Base implementation shared across different execution modes."""

import math
from abc import ABCMeta, abstractmethod
from typing import Any, Tuple

from JackFramework.Core.Graph import dataloader_selection, graph_selection
from JackFramework.SysBasic.show_handler import ShowHandler


class MetaMode(ShowHandler, metaclass=ABCMeta):
    def __init__(self, args: object, user_inference_func: Any,
                 is_training: bool = True) -> None:
        super().__init__()
        self._args = args
        self._is_training = is_training
        self._user_inference_func = user_inference_func
        self._graph = None
        self._data_manager = None
        self._training_iteration, self._val_iteration = self.__estimate_iteration_counts()

    # ------------------------------------------------------------------
    @property
    def args(self) -> object:
        return self._args

    def __estimate_iteration_counts(self) -> Tuple[int, int]:
        batch_size = max(self._args.batchSize, 1)
        world_size = max(self._args.gpu, 1) if getattr(self._args, 'dist', False) else 1
        samples_per_step = batch_size * world_size

        def _calc(images: int) -> int:
            if images <= 0:
                return 0
            return math.ceil(images * self._args.sampleNum / samples_per_step)

        return _calc(self._args.imgNum), _calc(self._args.valImgNum)

    # ------------------------------------------------------------------
    def _init_data_model_handler(self, rank: int = None) -> None:
        self.set_rank(rank)
        self.reinit_log_tensorboard_handler(self._args)
        jf_result = self._user_inference_func(self._args)
        if not isinstance(jf_result, (tuple, list)) or len(jf_result) != 2:
            raise ValueError('User interface must return a (model_handler, data_handler) tuple.')
        jf_model, jf_dataloader = jf_result
        if jf_model is None or jf_dataloader is None:
            raise ValueError('Model or dataloader handler returned by user interface is None.')

        self._graph = graph_selection(self._args, jf_model)
        self._data_manager = dataloader_selection(self._args, jf_dataloader)

    # ------------------------------------------------------------------
    def _get_img_id(self, iteration: int) -> int:
        if self.rank is None:
            return iteration
        per_rank = self._args.batchSize * max(self._args.gpu, 1)
        return self.rank + iteration * per_rank

    def _save_result(self, iteration: int, outputs_data: list, supplement: list) -> None:
        if self._data_manager is None:
            raise RuntimeError('Data manager has not been initialised.')
        img_id = self._get_img_id(iteration)
        self._data_manager.user_save_result(outputs_data, supplement, img_id)

    # ------------------------------------------------------------------
    @ShowHandler.show_method
    def _save_model(self, epoch: int) -> None:
        if self._graph is None:
            return
        if self._args.auto_save_num <= 0:
            return
        if (epoch + 1) % self._args.auto_save_num == 0:
            self._graph.save_model(epoch)

    @ShowHandler.show_method
    def _write_epoch_log(self, epoch: int) -> None:
        if self._data_manager is None or self._graph is None:
            return
        self._data_manager.user_show_training_info(
            epoch,
            self._graph.ave_tower_loss,
            self._graph.ave_tower_acc,
            self.duration(),
            self._is_training,
        )

    # ------------------------------------------------------------------
    def set_training_iteration(self, iteration: int) -> None:
        self._training_iteration = iteration

    def set_val_iteration(self, iteration: int) -> None:
        self._val_iteration = iteration

    @abstractmethod
    def exec(self, rank: int = None) -> None:
        """Execute the mode for the given distributed rank."""
