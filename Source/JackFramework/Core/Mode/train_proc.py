# -*- coding: utf-8 -*-
import torch.distributed as dist

from JackFramework.SysBasic.loghander import LogHandler as log
from JackFramework.SysBasic.show_handler import ShowHandler

from ._meta_mode import MetaMode


class TrainProc(MetaMode):
    """docstring for Executor"""

    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = True) -> object:
        super().__init__(args, user_inference_func, is_training)
        self.__args = args

    def __init_training_para(self, epoch: int, is_training: bool = True) -> tuple:
        total_iteration, off_set = 0, 1
        self._graph.init_result()
        self._graph.set_model_mode(is_training)
        dataloader = self._data_manager.get_dataloader(is_training)
        self._data_manager.set_epoch(epoch, is_training)
        self._graph.user_pretreatment(epoch)
        return total_iteration, off_set, dataloader

    def __training_data_proc(self, batch_data: list,
                             total_iteration: int, is_training: bool) -> tuple:
        input_data, output_data = self._data_manager.user_split_data(batch_data, True)
        self._graph.exec(input_data, output_data, is_training)
        self._graph.cal_tower_loss_acc(total_iteration)

    def __train_proc(self, epoch: int, training_iteration: int,
                     bar_info: str, is_training: bool = True) -> None:
        total_iteration, off_set, dataloader = self.__init_training_para(epoch, is_training)
        self.init_show_setting(training_iteration, bar_info)

        for iteration, batch_data in enumerate(dataloader):
            total_iteration = iteration + off_set
            self.__training_data_proc(batch_data, total_iteration, is_training)
            self._show_iteration_result(total_iteration, training_iteration, epoch)

        self._show_epoch_result(epoch, total_iteration, training_iteration, bar_info)
        self._adjust_lr_scheduler_and_post_proc(epoch, is_training)
        return total_iteration

    def __executor_training_proc(self, epoch: int) -> None:
        if self._training_iteration > 0:
            total_iteration = self.__train_proc(epoch, self._training_iteration, "Train", True)
            self.set_training_iteration(total_iteration)

    def __executor_val_proc(self, epoch: int) -> None:
        if self._val_iteration > 0:
            total_iteration = self.__train_proc(epoch, self._val_iteration, "Val", False)
            self.set_val_iteration(total_iteration)

    def _adjust_lr_scheduler_and_post_proc(self, epoch: int, is_training: bool) -> None:
        if is_training:
            self._graph.adjust_lr_scheduler(self._graph.ave_tower_loss)
        self._graph.user_postprocess(epoch, self._graph.ave_tower_loss, self._graph.ave_tower_acc)

    @ShowHandler.show_method
    def _show_iteration_result(self, total_iteration: int,
                               training_iteration: int, epoch: int):
        self.calculate_ave_runtime(total_iteration, training_iteration)
        info_str = self._data_manager.user_show_intermediate_result(
            epoch, self._graph.ave_tower_loss, self._graph.ave_tower_acc)
        self.update_show_bar(info_str)

    @ShowHandler.show_method
    def _show_epoch_result(self, epoch: int, total_iteration: int,
                           training_iteration: int, bar_info: str) -> None:
        self.stop_show_setting()
        self._write_epoch_log(epoch)
        self.write_tensorboard(epoch, self._graph.ave_tower_loss,
                               self._graph.ave_tower_acc, bar_info)
        if total_iteration != training_iteration:
            log.warning("The input images numbers is different the number of datasets!")

    def __training_post_proc(self) -> None:
        self._graph.cleanup()
        log.info("Finish training process!")

    def __training_loop(self) -> None:
        log.info("Start iteration!")
        for epoch in range(self.__args.maxEpochs):
            self.__executor_training_proc(epoch)
            self.__executor_val_proc(epoch)
            if self.__args.dist:
                dist.barrier()
            self._save_model(epoch)

    def __preparation_proc(self) -> None:
        self._graph.restore_model()

    def exec(self, rank: object = None) -> None:
        self._init_datahandler_modelhandler(rank)
        log.info("Start the training process!")
        self.__preparation_proc()
        self.__training_loop()
        self.__training_post_proc()
