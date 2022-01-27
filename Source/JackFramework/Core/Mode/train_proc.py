# -*- coding: utf-8 -*-
import time
import torch.distributed as dist

from JackFramework.SysBasic.loghander import LogHandler as log
from .meta_mode import MetaMode


class TrainProc(MetaMode):
    """docstring for Executor"""

    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = True) -> object:
        super().__init__(args, user_inference_func, is_training)
        self.__args = args

    def __init_training_para(self, epoch: int, rank: object, is_training: bool = True) -> tuple:
        total_iteration, off_set = 0, 1
        graph, data_manager = self._get_graph_and_data_manager
        graph.init_result()
        graph.set_model_mode(is_training)
        dataloader = data_manager.get_dataloader(is_training)
        data_manager.set_epoch(epoch, is_training)
        graph.pretreatment(epoch, rank)

        return total_iteration, off_set, dataloader

    def __training_data_proc(self, batch_data: list,
                             total_iteration: int, is_training: bool) -> tuple:
        graph, data_manager = self._get_graph_and_data_manager
        input_data, output_data = data_manager.split_data(batch_data, True)
        graph.exec(input_data, output_data, is_training)
        graph.cal_tower_loss_acc(total_iteration)

    def __train_proc(self, epoch: int, training_iteration: int,
                     bar_info: str, rank: object, is_training: bool = True) -> None:
        total_iteration, off_set, dataloader = self.__init_training_para(epoch, rank, is_training)
        self._init_show_setting(training_iteration, bar_info)

        for iteration, batch_data in enumerate(dataloader):
            total_iteration = iteration + off_set
            self.__training_data_proc(batch_data, total_iteration, is_training)
            self._show_iteration_result(rank, total_iteration, training_iteration, epoch)

        self._show_epoch_result(rank, epoch, total_iteration,
                                training_iteration, bar_info)
        self._adjust_lr_scheduler_and_post_proc(epoch, rank, is_training)
        return total_iteration

    def __executor_training_proc(self, epoch: int, rank: object) -> None:
        if self._training_iteration > 0:
            total_iteration = self.__train_proc(
                epoch, self._training_iteration, "Train", rank, True)
            self.set_training_iteration(total_iteration)

    def __executor_val_proc(self, epoch: int, rank: object) -> None:
        if self._val_iteration > 0:
            total_iteration = self.__train_proc(
                epoch, self._val_iteration, "Val", rank, False)
            self.set_val_iteration(total_iteration)

    def _adjust_lr_scheduler_and_post_proc(self, epoch: int, rank: object,
                                           is_training: bool) -> None:
        assert self._graph is not None
        graph = self._graph
        if is_training:
            graph.adjust_lr_scheduler(graph.ave_tower_loss, rank)
        graph.postprocess(epoch, rank, graph.ave_tower_loss, graph.ave_tower_acc)

    def _show_iteration_result(self, rank: object, total_iteration: int,
                               training_iteration: int, epoch: int):
        graph, data_manager = self._get_graph_and_data_manager
        if rank == MetaMode.DEFAULT_RANK_ID or rank is None:
            self._calculate_ave_runtime(total_iteration, training_iteration)
            info_str = data_manager.show_intermediate_result(
                epoch, graph.ave_tower_loss, graph.ave_tower_acc)
            self._update_show_bar(info_str)

    def _show_epoch_result(self, rank: object, epoch: int, total_iteration: int,
                           training_iteration: int, bar_info: str) -> None:
        if rank == MetaMode.DEFAULT_RANK_ID or rank is None:
            self._stop_show_setting()
            self._write_epoch_log(rank, epoch)
            self._write_tensorboard(rank, epoch, bar_info)

            if total_iteration != training_iteration:
                log.warning("The input images numbers is different the number of datasets!")

    def __traning_post_proc(self) -> None:
        graph = self._graph
        graph.cleanup()
        log.info("Finish training process!")

    def exec(self, rank: object = None) -> None:
        self._init_datahandler_modelhandler(rank)
        log.info("Start the training process!")
        args = self.__args
        graph = self._graph
        graph.restore_model(rank)

        log.info("Start iteration!")
        for epoch in range(args.maxEpochs):
            self.__executor_training_proc(epoch, rank)
            self.__executor_val_proc(epoch, rank)
            if args.dist:
                dist.barrier()
            self._save_model(epoch, rank)

        self.__traning_post_proc()
