# -*- coding: utf-8 -*-
import math
import time
import torch.distributed as dist

from .build_training_graph import BuildGraph
from .data_handler_manager import DataHandlerManager

from JackFramework.SysBasic.loghander import LogHandler as log
from JackFramework.SysBasic.processbar import ShowProcess
from JackFramework.FileHandler.tensorboard_handler import TensorboardHandler


class Executor(object):
    """docstring for Executor"""
    DEFAULT_RANK_ID = 0

    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = True) -> object:
        super().__init__()
        self.__args = args
        self.__is_training = is_training
        self.__user_inference_func = user_inference_func

        self.__data_manager = None
        self.__graph = None
        self.__tensorboard_handler = None

        if args.dist:
            self.__training_iteration = math.ceil(
                args.imgNum * args.sampleNum / (args.batchSize * args.gpu))
            self.__val_iteration = math.ceil(
                args.valImgNum * args.sampleNum / (args.batchSize * args.gpu))
        else:
            self.__training_iteration = math.ceil(args.imgNum * args.sampleNum / args.batchSize)
            self.__val_iteration = math.ceil(args.valImgNum * args.sampleNum / args.batchSize)

    def __init_datahandler_modelhandler(self, rank: object) -> object:
        args = self.__args

        if not args.dist:
            self.__tensorboard_handler = TensorboardHandler(args)
        elif rank == Executor.DEFAULT_RANK_ID:
            # dist reinit log
            log().init_log(args.outputDir, args.pretrain)
            log().info("LogHandler is reinitialized!")
            self.__tensorboard_handler = TensorboardHandler(args)

        model, dataloader = self.__user_inference_func(self.__args)
        assert model is not None and dataloader is not None
        graph = BuildGraph(self.__args, model, rank)
        data_manager = DataHandlerManager(self.__args, dataloader)

        return data_manager, graph

    def __init_training_para(self, epoch: int, rank: object, is_training: bool = True) -> tuple:
        total_iteration = 0
        off_set = 1
        tower_loss = None
        tower_acc = None
        ave_tower_acc = None
        ave_tower_loss = None
        graph, data_manager = self.__get_graph_and_data_manager()

        graph.set_model_mode(is_training)
        dataloader = data_manager.get_dataloader(is_training)
        data_manager.set_epoch(epoch, is_training)
        graph.pretreatment(epoch, rank)

        return total_iteration, off_set, dataloader, tower_loss,\
            tower_acc, ave_tower_acc, ave_tower_loss

    @staticmethod
    def __calculate_ave_runtime(start_time: object, end_time: object,
                                total_iteration: int, training_iteration: int) -> tuple:
        duration = (end_time - start_time) / (total_iteration)
        rest_time = (training_iteration - total_iteration) * duration
        return duration, rest_time

    def __training_data_proc(self, batch_data: list, tower_loss: list,
                             tower_acc: list, total_iteration: int, is_training: bool) -> tuple:
        graph, data_manager = self.__get_graph_and_data_manager()
        input_data, output_data = data_manager.split_data(batch_data, True)
        tower_loss_iteration, tower_acc_iteration = graph.train_proc(
            input_data, output_data, is_training)
        tower_loss, tower_acc, ave_tower_loss, ave_tower_acc = graph.cal_tower_loss_acc(
            tower_loss, tower_acc, tower_loss_iteration, tower_acc_iteration, total_iteration)
        return tower_loss, tower_acc, ave_tower_loss, ave_tower_acc

    def __testing_data_proc(self, batch_data: list) -> tuple:
        graph, data_manager = self.__get_graph_and_data_manager()
        input_data, supplement = data_manager.split_data(batch_data, False)
        outputs_data = graph.test_model(input_data)
        return outputs_data, supplement

    @staticmethod
    def __init_show_setting(training_iteration: int, bar_info: str) -> tuple:
        process_bar = ShowProcess(training_iteration, bar_info)
        start_time = time.time()
        return process_bar, start_time

    def __show_iteration_result(self, rank: object, process_bar: object, start_time: object,
                                total_iteration: int, training_iteration: int, epoch: int,
                                ave_tower_loss: list, ave_tower_acc: list):
        _, data_manager = self.__get_graph_and_data_manager()
        if rank == Executor.DEFAULT_RANK_ID or rank is None:
            duration, rest_time = self.__calculate_ave_runtime(
                start_time, time.time(), total_iteration, training_iteration)
            info_str = data_manager.show_intermediate_result(
                epoch, ave_tower_loss, ave_tower_acc)
            process_bar.show_process(show_info=info_str, rest_time=rest_time, duration=duration)

    def __show_epoch_result(self, rank: object, process_bar: object,
                            start_time: object, epoch: int, ave_tower_loss: list,
                            ave_tower_acc: int, total_iteration: int,
                            training_iteration: int, bar_info: str, is_training: bool) -> None:
        _, data_manager = self.__get_graph_and_data_manager()
        if rank == Executor.DEFAULT_RANK_ID or rank is None:
            process_bar.close()
            duration = time.time() - start_time
            data_manager.show_training_info(epoch, ave_tower_loss, ave_tower_acc, duration, is_training)
            self.__tensorboard_handler.write_data(epoch, ave_tower_loss, ave_tower_acc, bar_info)

            if total_iteration != training_iteration:
                log.warning("The input images numbers is different the number of datasets!")

    def __show_testing_proc(self, rank: object, start_time: object,
                            process_bar: object, total_iteration: int) -> None:
        if rank == Executor.DEFAULT_RANK_ID or rank is None:
            duration, rest_time = self.__calculate_ave_runtime(
                start_time, time.time(), total_iteration, self.__training_iteration)
            process_bar.show_process(rest_time=rest_time, duration=duration)

    def __adjust_lr_scheduler_and_post_proc(self, epoch: int, rank: object,
                                            ave_tower_loss: float, ave_tower_acc: float,
                                            is_training: bool) -> None:
        graph, _ = self.__get_graph_and_data_manager()
        if is_training:
            graph.adjust_lr_scheduler(ave_tower_loss, rank)
        graph.postprocess(epoch, rank, ave_tower_loss, ave_tower_acc)

    def __train_proc(self, epoch: int, training_iteration: int,
                     bar_info: str, rank: object, is_training: bool = True) -> None:
        total_iteration, off_set, dataloader, tower_loss, tower_acc, ave_tower_acc, \
            ave_tower_loss = self.__init_training_para(epoch, rank, is_training)
        process_bar, start_time = self.__init_show_setting(training_iteration, bar_info)

        for iteration, batch_data in enumerate(dataloader):
            total_iteration = iteration + off_set
            tower_loss, tower_acc, ave_tower_loss, ave_tower_acc = self.__training_data_proc(
                batch_data, tower_loss, tower_acc, total_iteration, is_training)
            self.__show_iteration_result(rank, process_bar, start_time, total_iteration,
                                         training_iteration, epoch, ave_tower_loss,
                                         ave_tower_acc)

        self.__show_epoch_result(rank, process_bar, start_time, epoch,
                                 ave_tower_loss, ave_tower_acc, total_iteration,
                                 training_iteration, bar_info, is_training)
        self.__adjust_lr_scheduler_and_post_proc(epoch, rank, ave_tower_loss,
                                                 ave_tower_acc, is_training)

        return total_iteration

    def __get_graph_and_data_manager(self):
        return self.__graph, self.__data_manager

    def __get_img_id(self, iteration: int, rank: object) -> int:
        args = self.__args
        if rank is None:
            return iteration
        else:
            return rank + iteration * (args.batchSize * args.gpu)

    def __executor_training_proc(self, epoch: int, rank: object) -> None:
        if self.__training_iteration > 0:
            self.__training_iteration = self.__train_proc(
                epoch, self.__training_iteration, "Train", rank, True)

    def __executor_val_proc(self, epoch: int, rank: object) -> None:
        if self.__val_iteration > 0:
            self.__val_iteration = self.__train_proc(
                epoch, self.__val_iteration, "Val", rank, False)

    def __save_model(self, epoch: int, rank: object) -> None:
        args = self.__args
        graph, _ = self.__get_graph_and_data_manager()
        off_set = 1

        if (epoch + off_set) % args.auto_save_num == 0 and (
                rank == Executor.DEFAULT_RANK_ID or rank is None):
            graph.save_model(epoch)

    def __save_result(self, iteration: int, rank: object,
                      outputs_data: list, supplement: list):
        _, data_manager = self.__get_graph_and_data_manager()
        img_id = self.__get_img_id(iteration, rank)
        data_manager.save_result(outputs_data, supplement, img_id)

    def __traning_post_proc(self) -> None:
        graph, _ = self.__get_graph_and_data_manager()
        graph.cleanup()
        log.info("Finish training process!")

    def __testing_post_proc(self, rank: object, process_bar: object) -> None:
        if rank == Executor.DEFAULT_RANK_ID or rank is None:
            process_bar.close()
            log.info("Finish testing process!")

    def init_datahandler_modelhandler(self, rank: object) -> None:
        self.__data_manager, self.__graph = self.__init_datahandler_modelhandler(rank)

    def train(self, rank: object = None) -> None:
        self.init_datahandler_modelhandler(rank)
        log.info("Start the training process!")
        args = self.__args
        graph, _ = self.__get_graph_and_data_manager()
        graph.restore_model(rank)

        log.info("Start iteration!")
        for epoch in range(args.maxEpochs):
            self.__executor_training_proc(epoch, rank)
            self.__executor_val_proc(epoch, rank)
            if args.dist:
                dist.barrier()

        self.__save_model(epoch, rank)
        self.__traning_post_proc()

    def test(self, rank: object = None) -> None:
        self.init_datahandler_modelhandler(rank)
        log.info("Start the testing process!")
        graph, _ = self.__get_graph_and_data_manager()
        graph.restore_model(rank)

        total_iteration, off_set, dataloader, _, _, _, _ = self.__init_training_para(0, rank, True)
        graph.set_model_mode(False)
        process_bar, start_time = self.__init_show_setting(self.__training_iteration, "Test")

        log.info("Start testing iteration!")
        for iteration, batch_data in enumerate(dataloader):
            total_iteration = iteration + off_set
            outputs_data, supplement = self.__testing_data_proc(batch_data)
            self.__save_result(iteration, rank, outputs_data, supplement)
            self.__show_testing_proc(rank, start_time, process_bar, total_iteration)

        graph.postprocess(0, rank)
        self.__testing_post_proc(rank, process_bar)
