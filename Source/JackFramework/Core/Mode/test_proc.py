# -*- coding: utf-8 -*-
from JackFramework.SysBasic.show_handler import ShowHandler
from JackFramework.SysBasic.loghander import LogHandler as log
from .meta_mode import MetaMode


class TestProc(MetaMode):
    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = True) -> object:
        super().__init__(args, user_inference_func, is_training)
        self.__args = args

    def __init_testing_para(self, epoch: int, is_training: bool = True) -> tuple:
        total_iteration, off_set = 0, 1
        graph, data_manager = self._get_graph_and_data_manager
        graph.set_model_mode(is_training)
        dataloader = data_manager.get_dataloader(is_training)
        data_manager.set_epoch(epoch, is_training)
        graph.pretreatment(epoch)
        return total_iteration, off_set, dataloader

    def __testing_data_proc(self, batch_data: list) -> tuple:
        graph, data_manager = self._get_graph_and_data_manager
        input_data, supplement = data_manager.split_data(batch_data, False)
        outputs_data = graph.exec(input_data, None)
        return outputs_data, supplement

    @ShowHandler.show_method
    def _show_testing_proc(self, total_iteration: int) -> None:
        self.calculate_ave_runtime(total_iteration, self._training_iteration)
        self.update_show_bar('')

    @ShowHandler.show_method
    def _testing_post_proc(self) -> None:
        self.stop_show_setting()
        log.info("Finish testing process!")

    def exec(self, rank: object = None) -> None:
        self._init_datahandler_modelhandler(rank)
        log.info("Start the testing process!")
        graph = self._graph
        graph.restore_model()
        total_iteration, off_set, dataloader = self.__init_testing_para(0, True)
        graph.set_model_mode(False)
        self.init_show_setting(self._training_iteration, "Test")

        log.info("Start testing iteration!")
        for iteration, batch_data in enumerate(dataloader):
            total_iteration = iteration + off_set
            outputs_data, supplement = self.__testing_data_proc(batch_data)
            self._save_result(iteration, outputs_data, supplement)
            self._show_testing_proc(total_iteration)

        graph.postprocess(0)
        self._testing_post_proc()
