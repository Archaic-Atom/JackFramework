# -*- coding: utf-8 -*-
import torch

from JackFramework.SysBasic.loghander import LogHandler as log


class DataHandlerManager(object):
    """docstring for ClassName"""

    def __init__(self, args: object, jf_datahandler: object) -> None:
        super().__init__()
        self.__jf_datahandler = jf_datahandler
        self.__args = args
        self.__training_dataloader, self.__val_dataloader, self.__training_sampler,\
            self.__val_sampler = self.__init_dataloader()

    def __init_dataloader(self) -> None:
        log.info("Begin loading the training dataset")
        args = self.__args

        training_dataloader = None
        val_dataloader = None
        training_sampler = None
        val_sampler = None

        is_training = True
        if args.mode == 'test':
            is_training = False

        if args.imgNum > 0:
            tranining_dataset = self.__jf_datahandler.get_train_dataset(
                self.__args.trainListPath, is_training)

            dataloader_shuffle = is_training
            if args.dist:
                training_sampler = torch.utils.data.distributed.DistributedSampler(
                    tranining_dataset, shuffle=is_training)
                dataloader_shuffle = False
                # self.__args.dataloaderNum = 0

            training_dataloader = torch.utils.data.DataLoader(
                tranining_dataset,
                batch_size=self.__args.batchSize,
                shuffle=dataloader_shuffle,
                num_workers=self.__args.dataloaderNum,
                pin_memory=True,
                sampler=training_sampler
            )
        else:
            log.error("The training images is 0")

        log.info("Begin loading the val dataset")
        if args.valImgNum > 0:
            val_dataset = self.__jf_datahandler.get_val_dataset(self.__args.valListPath)

            if args.dist:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    tranining_dataset, shuffle=False)
                self.__args.dataloaderNum = 0

            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.__args.batchSize,
                num_workers=self.__args.dataloaderNum,
                pin_memory=True,
                sampler=val_sampler
            )
        else:
            log.warning("The val images is 0")

        log.info("Finish constrcuting the dataloader")

        return training_dataloader, val_dataloader, training_sampler, val_sampler

    @property
    def training_dataloader(self) -> object:
        return self.__training_dataloader

    @property
    def val_dataloader(self) -> object:
        return self.__val_dataloader

    def get_dataloader(self, is_traning: bool) -> object:
        if is_traning:
            return self.training_dataloader
        return self.val_dataloader

    def set_epoch(self, epoch: int, is_traning: bool) -> None:
        args = self.__args
        if args.dist:
            if is_traning:
                self.__training_sampler.set_epoch(epoch)
            else:
                self.__val_sampler.set_epoch(epoch)

    def split_data(self, batch_data: tuple, is_training: bool) -> object:
        return self.__jf_datahandler.split_data(batch_data, is_training)

    def show_training_info(self, epoch: int, loss:
                           list, acc: list, duration: float,
                           is_training: bool) -> None:
        assert self.__jf_datahandler is not None

        if is_training:
            self.__jf_datahandler.show_train_result(epoch, loss, acc, duration)
        else:
            self.__jf_datahandler.show_val_result(epoch, loss, acc, duration)

    def show_intermediate_result(self, epoch: int,
                                 loss: list, acc: list) -> str:
        return self.__jf_datahandler.show_intermediate_result(epoch, loss, acc)

    def save_result(self, output_data: list, supplement: list, img_id: int):
        for idx, output_item in enumerate(output_data):
            self.__jf_datahandler.save_result(output_item, supplement, img_id, idx)
