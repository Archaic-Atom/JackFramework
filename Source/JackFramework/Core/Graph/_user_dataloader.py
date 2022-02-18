# -*- coding: utf-8 -*-
from JackFramework.UserTemplate import DataHandlerTemplate


class UserDataloader(object):
    def __init__(self, args: object, jf_datahandler: object) -> None:
        super().__init__()
        assert isinstance(jf_datahandler, DataHandlerTemplate)
        self.__args = args
        self.__jf_datahandler = jf_datahandler

    def user_get_train_dataset(self, is_training: bool) -> object:
        return self.__jf_datahandler.get_train_dataset(self.__args.trainListPath, is_training)

    def user_get_val_dataset(self) -> object:
        return self.__jf_datahandler.get_val_dataset(self.__args.valListPath)

    def user_split_data(self, batch_data: tuple, is_training: bool) -> object:
        return self.__jf_datahandler.split_data(batch_data, is_training)

    def user_show_training_info(self, epoch: int, loss: list,
                                acc: list, duration: float, is_training: bool) -> None:
        if is_training:
            self.__jf_datahandler.show_train_result(epoch, loss, acc, duration)
        else:
            self.__jf_datahandler.show_val_result(epoch, loss, acc, duration)

    def user_show_intermediate_result(self, epoch: int, loss: list, acc: list) -> str:
        return self.__jf_datahandler.show_intermediate_result(epoch, loss, acc)

    def user_save_result(self, output_data: list, supplement: list, img_id: int) -> None:
        for idx, output_item in enumerate(output_data):
            self.__jf_datahandler.save_result(output_item, supplement, img_id, idx)

    def user_load_test_data(self, cmd: str) -> tuple:
        return self.__jf_datahandler.load_test_data(cmd)

    def user_save_test_data(self, output_data: list, supplement: list, cmd: str) -> None:
        for idx, output_item in enumerate(output_data):
            return self.__jf_datahandler.save_test_data(output_item, supplement, cmd, idx)
