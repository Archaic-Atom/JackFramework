# -*- coding: utf-8 -*-

DEFAULT_MAX_DECIMAL_PLACES = 6
DEFAULT_MIN_DECIMAL_PLACES = 2


class ResultStr(object):
    """docstring for ResultStr"""

    def __init__(self, arg=None):
        super().__init__()
        self.__arg = arg

    def training_result_str(self, epoch: int, loss: list, acc: list,
                            duration: float, training=True) -> str:
        loss_str = self.loss2str(loss,
                                 decimal_places=DEFAULT_MAX_DECIMAL_PLACES)
        acc_str = self.acc2str(acc,
                               decimal_places=DEFAULT_MAX_DECIMAL_PLACES)

        training_state = ""
        if training:
            training_state = "[TrainProcess] "
        else:
            training_state = "[ValProcess] "

        info_str = training_state + "e: " + str(epoch) + ', ' +\
            loss_str + ', ' + acc_str + ' (%.3f s/epoch)' % duration

        return info_str

    def training_intermediate_result(self, epoch: int, loss: list,
                                     acc: list)->str:
        loss_str = self.loss2str(loss, decimal_places=3)
        acc_str = self.acc2str(acc, decimal_places=3)

        info_str = 'e: ' + str(epoch) + ', ' +\
            loss_str + ', ' + acc_str

        return info_str

    def loss2str(self, loss: list, info_str: str = None,
                 decimal_places: int = DEFAULT_MIN_DECIMAL_PLACES)->str:
        if info_str is None:
            info_str = []
            info_str = self.__gen_info_str("l", len(loss))

        res = self.__data2str(loss, info_str, decimal_places)

        return res

    def acc2str(self, acc: list, info_str: str = None,
                decimal_places: int = DEFAULT_MIN_DECIMAL_PLACES)->str:
        if info_str is None:
            info_str = []
            info_str = self.__gen_info_str("a", len(acc))

        res = self.__data2str(acc, info_str, decimal_places)
        return res

    def __gen_info_str(self, info_str: str, num: int)-> str:
        res = []
        for i in range(num):
            res.append(info_str + str(i))
        return res

    def __data2str(self, data: list, info_str: list,
                   decimal_places: int) -> str:
        assert len(data) == len(info_str)
        res = ""
        char_interval = ", "
        for i in range(len(info_str)):
            res = res + info_str[i] + \
                (": %." + str(decimal_places) + "f") % data[i] + char_interval

        char_offset = len(char_interval)
        res = res[:len(res)-char_offset]
        return res
