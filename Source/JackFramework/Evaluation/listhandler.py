# -*- coding: utf-8 -*-
class ListHandler(object):
    """docstring for ListHandler"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def list_add(list_A: list, list_B: list) -> list:
        assert len(list_A) == len(list_B)
        return [item + list_B[i] for i, item in enumerate(list_A)]

    @staticmethod
    def list_div(list_A: list, num: float) -> list:
        return [item / num for _, item in enumerate(list_A)]

    @staticmethod
    def list_mean(list_A: list) -> list:
        return [item for _, item in enumerate(list_A)]

    @staticmethod
    def double_list_add(list_A: list, list_B: list = None) -> list:
        assert type(list_A) == list
        assert type(list_A[0]) == list
        if list_B is None:
            return list_A

        for i, item in enumerate(list_A):
            list_A[i] = ListHandler.list_add(item, list_B[i])

        return list_A

    @staticmethod
    def double_list_div(list_A: list, num: float) -> None:
        res = []
        for _, item in enumerate(list_A):
            tem_res = ListHandler.list_div(item, num)
            res.append(tem_res)

        return res
