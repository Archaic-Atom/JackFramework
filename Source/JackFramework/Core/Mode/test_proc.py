# -*- coding: utf-8 -*-
import time
from JackFramework.SysBasic.loghander import LogHandler as log
from .meta_mode import MetaMode


class TestProc(MetaMode):
    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = True) -> object:
        super().__init__(args, user_inference_func, is_training)
        self.__args = args

    def __init_testing_para(self, epoch: int, rank: object, is_training: bool = True) -> tuple:
        total_iteration, off_set = 0, 1
        graph, data_manager = self._get_graph_and_data_manager
        graph.set_model_mode(is_training)
        dataloader = data_manager.get_dataloader(is_training)
        data_manager.set_epoch(epoch, is_training)
        graph.pretreatment(epoch, rank)
        return total_iteration, off_set, dataloader

    def __testing_data_proc(self, batch_data: list) -> tuple:
        graph, data_manager = self._get_graph_and_data_manager
        input_data, supplement = data_manager.split_data(batch_data, False)
        outputs_data = graph.exec(input_data, None)
        return outputs_data, supplement

    def _show_testing_proc(self, rank: object, start_time: object,
                           process_bar: object, total_iteration: int) -> None:
        if rank == MetaMode.DEFAULT_RANK_ID or rank is None:
            duration, rest_time = self._calculate_ave_runtime(
                start_time, time.time(), total_iteration, self._training_iteration)
            process_bar.show_process(rest_time=rest_time, duration=duration)

    def _testing_post_proc(self, rank: object, process_bar: object) -> None:
        self._stop_show_setting(rank, process_bar)
        log.info("Finish testing process!")

    def exec(self, rank: object = None) -> None:
        self._init_datahandler_modelhandler(rank)
        log.info("Start the testing process!")
        graph = self._graph
        graph.restore_model(rank)

        total_iteration, off_set, dataloader = self.__init_testing_para(0, rank, True)
        graph.set_model_mode(False)
        process_bar, start_time = self._init_show_setting(rank, self._training_iteration, "Test")

        log.info("Start testing iteration!")
        for iteration, batch_data in enumerate(dataloader):
            total_iteration = iteration + off_set
            outputs_data, supplement = self.__testing_data_proc(batch_data)
            self._save_result(iteration, rank, outputs_data, supplement)
            self._show_testing_proc(rank, start_time, process_bar, total_iteration)

        graph.postprocess(0, rank)
        self._testing_post_proc(rank, process_bar)
