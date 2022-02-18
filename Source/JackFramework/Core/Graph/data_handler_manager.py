# -*- coding: utf-8 -*-
import torch
from JackFramework.SysBasic.loghander import LogHandler as log
from ._user_dataloader import UserDataloader


class DataHandlerManager(UserDataloader):
    """docstring for ClassName"""

    def __init__(self, args: object, jf_datahandler: object) -> None:
        super().__init__(args, jf_datahandler)
        self.__args = args
        self.__training_dataloader, self.__training_sampler = self.__check_training_dataloader()
        self.__val_dataloader, self.__val_sampler = self.__check_val_dataloader()

    @property
    def training_dataloader(self) -> object:
        return self.__training_dataloader

    @property
    def val_dataloader(self) -> object:
        return self.__val_dataloader

    def __get_sampler_shuffle(self, dataset: object, shuffle: bool) -> tuple:
        sampler = None
        if self.__args.dist:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle)
            shuffle = False
        return sampler, shuffle

    def __init_training_dataloader(self, is_training: bool) -> tuple:
        training_dataset = self.user_get_train_dataset(is_training)
        training_sampler, dataloader_shuffle = self.__get_sampler_shuffle(training_dataset,
                                                                          is_training)
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset, batch_size=self.__args.batchSize, shuffle=dataloader_shuffle,
            num_workers=self.__args.dataloaderNum, pin_memory=True, sampler=training_sampler,
            collate_fn=self.__collate_fn(training_dataset))
        return training_dataloader, training_sampler

    def __init_val_dataloader(self) -> tuple:
        val_dataset = self.user_get_val_dataset()
        val_sampler, _ = self.__get_sampler_shuffle(val_dataset, False)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.__args.batchSize, num_workers=self.__args.dataloaderNum,
            pin_memory=True, sampler=val_sampler, collate_fn=self.__collate_fn(val_dataset))
        return val_dataloader, val_sampler

    def __check_training_dataloader(self) -> tuple:
        log.info("Begin loading the training dataset")
        if self.__args.imgNum > 0:
            is_training = self.__args.mode != 'test'
            training_dataloader, training_sampler = self.__init_training_dataloader(is_training)
        else:
            log.warning("The training images is 0")
            training_dataloader, training_sampler = None, None
        log.info("Finish constructing the training dataloader")
        return training_dataloader, training_sampler

    def __check_val_dataloader(self) -> object:
        log.info("Begin loading the val dataset")
        if self.__args.valImgNum > 0:
            val_dataloader, val_sampler = self.__init_val_dataloader()
        else:
            log.warning("The val images is 0")
            val_dataloader, val_sampler = None, None
        log.info("Finish constructing the val dataloader")
        return val_dataloader, val_sampler

    def get_dataloader(self, is_training: bool) -> object:
        return self.training_dataloader if is_training else self.val_dataloader

    def set_epoch(self, epoch: int, is_training: bool) -> None:
        if self.__args.dist:
            if is_training:
                self.__training_sampler.set_epoch(epoch)
            else:
                self.__val_sampler.set_epoch(epoch)

    @staticmethod
    def __collate_fn(dataset: object) -> object:
        try:
            func = dataset.collate_fn
        except AttributeError:
            func = None
        return func
