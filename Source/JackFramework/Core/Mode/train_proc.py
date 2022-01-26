# -*- coding: utf-8 -*-
import math
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
        tower_loss, tower_acc, ave_tower_acc, ave_tower_loss = None, None, None, None
        graph, data_manager = self._get_graph_and_data_manager

        graph.set_model_mode(is_training)
        dataloader = data_manager.get_dataloader(is_training)
        data_manager.set_epoch(epoch, is_training)
        graph.pretreatment(epoch, rank)

        return total_iteration, off_set, dataloader, tower_loss,\
            tower_acc, ave_tower_acc, ave_tower_loss

    def __training_data_proc(self, batch_data: list, tower_loss: list,
                             tower_acc: list, total_iteration: int, is_training: bool) -> tuple:
        graph, data_manager = self._get_graph_and_data_manager
        input_data, output_data = data_manager.split_data(batch_data, True)
        tower_loss_iteration, tower_acc_iteration = graph.exec(input_data, output_data, is_training)
        tower_loss, tower_acc, ave_tower_loss, ave_tower_acc = graph.cal_tower_loss_acc(
            tower_loss, tower_acc, tower_loss_iteration, tower_acc_iteration, total_iteration)
        return tower_loss, tower_acc, ave_tower_loss, ave_tower_acc

    def __train_proc(self, epoch: int, training_iteration: int,
                     bar_info: str, rank: object, is_training: bool = True) -> None:
        total_iteration, off_set, dataloader, tower_loss, tower_acc, ave_tower_acc, \
            ave_tower_loss = self.__init_training_para(epoch, rank, is_training)
        process_bar, start_time = self._init_show_setting(rank, training_iteration, bar_info)

        for iteration, batch_data in enumerate(dataloader):
            total_iteration = iteration + off_set
            tower_loss, tower_acc, ave_tower_loss, ave_tower_acc = self.__training_data_proc(
                batch_data, tower_loss, tower_acc, total_iteration, is_training)
            self._show_iteration_result(rank, process_bar, start_time, total_iteration,
                                        training_iteration, epoch, ave_tower_loss,
                                        ave_tower_acc)

        self._show_epoch_result(rank, process_bar, start_time, epoch, ave_tower_loss,
                                ave_tower_acc, total_iteration, training_iteration,
                                bar_info, is_training)
        self._adjust_lr_scheduler_and_post_proc(epoch, rank, ave_tower_loss,
                                                ave_tower_acc, is_training)
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
                                           ave_tower_loss: float, ave_tower_acc: float,
                                           is_training: bool) -> None:
        assert self._graph is not None
        graph = self._graph
        if is_training:
            graph.adjust_lr_scheduler(ave_tower_loss, rank)
        graph.postprocess(epoch, rank, ave_tower_loss, ave_tower_acc)

    def _show_iteration_result(self, rank: object, process_bar: object, start_time: object,
                               total_iteration: int, training_iteration: int, epoch: int,
                               ave_tower_loss: list, ave_tower_acc: list):
        _, data_manager = self._get_graph_and_data_manager
        if rank == MetaMode.DEFAULT_RANK_ID or rank is None:
            duration, rest_time = self._calculate_ave_runtime(
                start_time, time.time(), total_iteration, training_iteration)
            info_str = data_manager.show_intermediate_result(
                epoch, ave_tower_loss, ave_tower_acc)
            process_bar.show_process(show_info=info_str, rest_time=rest_time, duration=duration)

    def _show_epoch_result(self, rank: object, process_bar: object,
                           start_time: object, epoch: int, ave_tower_loss: list,
                           ave_tower_acc: int, total_iteration: int,
                           training_iteration: int, bar_info: str, is_training: bool) -> None:
        _, data_manager = self._get_graph_and_data_manager
        if rank == MetaMode.DEFAULT_RANK_ID or rank is None:
            process_bar.close()
            duration = time.time() - start_time
            data_manager.show_training_info(
                epoch, ave_tower_loss, ave_tower_acc, duration, is_training)
            self._tensorboard_handler.write_data(epoch, ave_tower_loss, ave_tower_acc, bar_info)

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
