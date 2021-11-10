# -*- coding: utf-8 -*-
import torch
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

    def __init_training_model(self, epoch: int, graph: BuildGraph,
                              data_manager: DataHandlerManager,
                              is_training: bool = True) -> tuple:
        total_iteration = 0
        off_set = 1

        graph.set_model_mode(is_training)
        dataloader = data_manager.get_dataloader(is_training)
        data_manager.set_epoch(epoch, is_training)

        tower_loss = None
        tower_acc = None
        ave_tower_acc = None
        ave_tower_loss = None

        return total_iteration, off_set, dataloader, tower_loss,\
            tower_acc, ave_tower_acc, ave_tower_loss

    def __calculate_ave_runtime(self, start_time: object, end_time: object,
                                total_iteration: int, training_iteration: int) -> tuple:
        duration = (end_time - start_time) / (total_iteration)
        rest_time = (training_iteration - total_iteration) * duration
        return duration, rest_time

    def __train_proc(self, epoch: int, training_iteration: int,
                     bar_info: str, rank: object, is_training: bool = True) -> None:
        graph = self.__graph
        data_manager = self.__data_manager
        total_iteration, off_set, dataloader, tower_loss,\
            tower_acc, ave_tower_acc, ave_tower_loss = self.__init_training_model(
                epoch, graph, data_manager, is_training)

        process_bar = ShowProcess(training_iteration, bar_info)
        start_time = time.time()

        for iteration, batch_data in enumerate(dataloader):
            # iterate process
            total_iteration = iteration + off_set
            input_data, output_data = data_manager.split_data(batch_data, True)
            tower_loss_iteration, tower_acc_iteration = graph.train_proc(
                input_data, output_data, is_training)
            tower_loss, tower_acc, ave_tower_loss, ave_tower_acc = graph.cal_tower_loss_acc(
                tower_loss, tower_acc, tower_loss_iteration, tower_acc_iteration, total_iteration)

            if rank == Executor.DEFAULT_RANK_ID or rank is None:
                duration, rest_time = self.__calculate_ave_runtime(
                    start_time, time.time(), total_iteration, training_iteration)
                info_str = data_manager.show_intermediate_result(
                    epoch, ave_tower_loss, ave_tower_acc)
                process_bar.show_process(show_info=info_str, rest_time=rest_time, duration=duration)

        if rank == Executor.DEFAULT_RANK_ID or rank is None:
            process_bar.close()
            duration = time.time() - start_time
            data_manager.show_training_info(
                epoch, ave_tower_loss, ave_tower_acc, duration, is_training)
            self.__tensorboard_handler.write_data(epoch, ave_tower_loss, ave_tower_acc, bar_info)

            if total_iteration != training_iteration:
                log.warning("The input images numbers is different the number of datasets!")

        if is_training:
            graph.adjust_lr_scheduler(ave_tower_loss, rank)

        return total_iteration

    def __get_graph_and_data_manager(self):
        return self.__graph, self.__data_manager

    def __get_img_id(self, iteration: int, rank: object) -> int:
        args = self.__args
        img_id = None
        if rank is None:
            return iteration
        else:
            return rank + iteration * (args.batchSize * args.gpu)

    def init_datahandler_modelhandler(self, rank: object):
        self.__data_manager, self.__graph = self.__init_datahandler_modelhandler(rank)

    def train(self, rank: object = None) -> None:
        self.init_datahandler_modelhandler(rank)
        log.info("Start the training process!")

        args = self.__args
        graph, _ = self.__get_graph_and_data_manager()
        graph.restore_model(rank)
        off_set = 1

        log.info("Start iteration!")
        for epoch in range(args.maxEpochs):
            if self.__training_iteration > 0:
                self.__training_iteration = self.__train_proc(
                    epoch, self.__training_iteration, "Train", rank, True)

            if self.__val_iteration > 0:
                self.__val_iteration = self.__train_proc(
                    epoch, self.__val_iteration, "Val", rank, False)

            if args.dist:
                dist.barrier()

            if (epoch + off_set) % args.auto_save_num == 0 and (
                    rank == Executor.DEFAULT_RANK_ID or rank is None):
                graph.save_model(epoch)

        graph.cleanup()
        log.info("Finish training process!")

    def test(self, rank: object = None) -> None:
        self.init_datahandler_modelhandler(rank)
        log.info("Start the testing process!")

        graph, data_manager = self.__get_graph_and_data_manager()

        graph.restore_model(rank)
        graph.set_model_mode(False)

        total_iteration, off_set, dataloader, _,\
            _, _, _ = self.__init_training_model(
                0, graph, data_manager, True)

        start_time = time.time()
        process_bar = ShowProcess(self.__training_iteration, "Test")

        log.info("Start iteration!")
        for iteration, batch_data in enumerate(dataloader):
            total_iteration = iteration + off_set
            input_data, supplement = data_manager.split_data(batch_data, False)
            outputs_data = graph.test_model(input_data)

            img_id = self.__get_img_id(iteration, rank)
            data_manager.save_result(outputs_data, supplement, img_id)

            if rank == Executor.DEFAULT_RANK_ID or rank is None:
                duration, rest_time = self.__calculate_ave_runtime(
                    start_time, time.time(), total_iteration, self.__training_iteration)
                process_bar.show_process(rest_time=rest_time, duration=duration)

        if rank == Executor.DEFAULT_RANK_ID or rank is None:
            process_bar.close()
            log.info("Finish testing process!")
