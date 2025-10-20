# -*- coding: utf-8 -*-
from JackFramework.SysBasic.show_handler import ShowHandler
from JackFramework.SysBasic.log_handler import LogHandler as log

from ._meta_mode import MetaMode


class TestProc(MetaMode):
    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = False) -> None:
        super().__init__(args, user_inference_func, is_training)

    def __prepare_testing(self, epoch: int) -> tuple:
        if self._graph is None or self._data_manager is None:
            raise RuntimeError('Graph or data manager not initialised before testing loop.')

        self._graph.set_model_mode(True)
        dataloader = self._data_manager.get_dataloader(True)
        if dataloader is None:
            log.warning('No testing dataloader available; skipping test execution.')
            return 0, None

        self._data_manager.set_epoch(epoch, True)
        self._graph.set_model_mode(False)
        self._graph.user_pretreatment(epoch)
        return 0, dataloader

    def _testing_data_proc(self, batch_data: list) -> tuple:
        input_data, supplement = self._data_manager.user_split_data(batch_data, False)
        outputs_data = self._graph.exec(input_data, None)
        return outputs_data, supplement

    @ShowHandler.show_method
    def _show_testing_proc(self, total_iteration: int) -> None:
        self.calculate_ave_runtime(total_iteration, self._training_iteration)
        self.update_show_bar('')

    @ShowHandler.show_method
    def _testing_post_proc(self) -> None:
        # UI-only teardown/logging on default rank
        self.stop_show_setting()
        log.info('Finish testing process!')

    def __test_loop(self) -> None:
        total_iteration, dataloader = self.__prepare_testing(0)
        if dataloader is None:
            return

        self.init_show_setting(self._training_iteration, 'Test')
        log.info('Start testing iteration!')
        for iteration, batch_data in enumerate(dataloader, start=1):
            total_iteration = iteration
            outputs_data, supplement = self._testing_data_proc(batch_data)
            self._save_result(iteration - 1, outputs_data, supplement)
            self._show_testing_proc(total_iteration)
        self._graph.user_post_process(0)

    def __preparation_proc(self) -> None:
        self._graph.restore_model()

    def exec(self, rank: int = None) -> None:
        # Ensure distributed shutdown even if exceptions occur
        try:
            self._init_data_model_handler(rank)
            log.info('Start the testing process!')
            self.__preparation_proc()
            self.__test_loop()
        finally:
            # Cleanup must run on all ranks, not gated by show_method
            try:
                if self._graph is not None:
                    # Pre-cleanup barrier to align shutdown order
                    try:
                        import torch.distributed as dist  # local import to avoid test-only dep at import time
                        if getattr(self._args, 'dist', False) and dist.is_initialized():
                            try:
                                if hasattr(dist, 'monitored_barrier'):
                                    from datetime import timedelta
                                    dist.monitored_barrier(timeout=timedelta(seconds=60))
                                else:
                                    dist.barrier()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    self._graph.cleanup()
            except Exception:
                pass
            # UI/logging only on default rank
            try:
                self._testing_post_proc()
            except Exception:
                pass
            # No direct destroy here; centralized in DeviceManager.cleanup()
