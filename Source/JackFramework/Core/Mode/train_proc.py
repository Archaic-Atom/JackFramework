# -*- coding: utf-8 -*-
import torch.distributed as dist

from JackFramework.SysBasic.log_handler import LogHandler as log
from JackFramework.SysBasic.show_handler import ShowHandler

from ._meta_mode import MetaMode


class TrainProc(MetaMode):
    """Training/validation loop coordinator."""

    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = True) -> None:
        super().__init__(args, user_inference_func, is_training)

    # ------------------------------------------------------------------
    def __prepare_epoch(self, epoch: int, is_training: bool) -> tuple:
        if self._graph is None or self._data_manager is None:
            raise RuntimeError('Graph or data manager not initialised before training loop.')

        self._graph.init_result()
        self._graph.set_model_mode(is_training)
        dataloader = self._data_manager.get_dataloader(is_training)
        if dataloader is None:
            log.warning('No dataloader available for %s phase; skipping.',
                        'training' if is_training else 'validation')
            return 0, dataloader

        self._data_manager.set_epoch(epoch, is_training)
        self._graph.user_pretreatment(epoch)
        return 0, dataloader

    def __process_batch(self, batch_data: list, total_iteration: int, is_training: bool) -> None:
        input_data, output_data = self._data_manager.user_split_data(batch_data, True)
        self._graph.exec(input_data, output_data, is_training)
        self._graph.cal_tower_loss_acc(total_iteration)

    def __run_epoch(self, epoch: int, planned_iterations: int,
                     bar_info: str, is_training: bool) -> int:
        total_iteration, dataloader = self.__prepare_epoch(epoch, is_training)
        if dataloader is None:
            return 0

        self.init_show_setting(planned_iterations, bar_info)
        for iteration, batch_data in enumerate(dataloader, start=1):
            total_iteration = iteration
            self.__process_batch(batch_data, total_iteration, is_training)
            self._show_iteration_result(total_iteration, planned_iterations, epoch)

        self._show_epoch_result(epoch, total_iteration, planned_iterations, bar_info)
        self._adjust_lr_scheduler_and_post_proc(epoch, is_training)
        return total_iteration

    def __executor_training_proc(self, epoch: int) -> None:
        if self._training_iteration > 0:
            total_iteration = self.__run_epoch(epoch, self._training_iteration, 'Train', True)
            self.set_training_iteration(total_iteration)

    def __executor_val_proc(self, epoch: int) -> None:
        if self._val_iteration > 0:
            total_iteration = self.__run_epoch(epoch, self._val_iteration, 'Val', False)
            self.set_val_iteration(total_iteration)

    def _adjust_lr_scheduler_and_post_proc(self, epoch: int, is_training: bool) -> None:
        if is_training:
            self._graph.adjust_lr_scheduler(self._graph.ave_tower_loss)
        self._graph.user_post_process(epoch, self._graph.ave_tower_loss, self._graph.ave_tower_acc)

    @ShowHandler.show_method
    def _show_iteration_result(self, total_iteration: int,
                               training_iteration: int, epoch: int) -> None:
        self.calculate_ave_runtime(total_iteration, training_iteration)
        info_str = self._data_manager.user_show_intermediate_result(
            epoch, self._graph.ave_tower_loss, self._graph.ave_tower_acc)
        self.update_show_bar(info_str)

    @ShowHandler.show_method
    def _show_epoch_result(self, epoch: int, total_iteration: int,
                           planned_iteration: int, bar_info: str) -> None:
        epoch_duration = self.duration()
        self.stop_show_setting()
        self._write_epoch_log(epoch, epoch_duration)
        self.write_tensorboard(epoch, self._graph.ave_tower_loss,
                               self._graph.ave_tower_acc, bar_info)
        if planned_iteration and total_iteration != planned_iteration:
            log.warning('Processed iterations (%s) differ from planned total (%s).',
                        total_iteration, planned_iteration)

    def __training_post_proc(self) -> None:
        # Align all ranks before destroying the process group to avoid
        # destructor ordering issues on exit.
        try:
            if getattr(self._args, 'dist', False) and dist.is_initialized():
                try:
                    if hasattr(dist, 'monitored_barrier'):
                        from datetime import timedelta
                        dist.monitored_barrier(timeout=timedelta(seconds=60))
                    else:
                        dist.barrier()
                except Exception:
                    pass
        finally:
            self._graph.cleanup()
        log.info('Finish training process!')

    def __training_loop(self) -> None:
        log.info('Start iteration!')
        for epoch in range(self._args.maxEpochs):
            self.__executor_training_proc(epoch)
            self.__executor_val_proc(epoch)
            if getattr(self._args, 'dist', False):
                barrier_kwargs = {'device_ids': [self.rank]} if self.rank is not None else {}
                try:
                    dist.barrier(**barrier_kwargs)
                except TypeError:
                    dist.barrier()
            self._save_model(epoch)

    def __preparation_proc(self) -> None:
        self._graph.restore_model()

    def exec(self, rank: int = None) -> None:
        # Ensure distributed shutdown even if exceptions occur
        try:
            self._init_data_model_handler(rank)
            log.info('Start the training process!')
            self.__preparation_proc()
            self.__training_loop()
        finally:
            # Preferred path: run normal post-processing (includes model/resource cleanup)
            try:
                self.__training_post_proc()
            except Exception:
                pass
