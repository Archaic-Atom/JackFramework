# -*- coding: utf-8 -*-
import numpy as np


class ListHandler(object):
    """docstring for ListHandler"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def list_add(list_A: list, list_B: list) -> list:
        assert len(list_A) == len(list_B)
        res = []

        for i, item in enumerate(list_A):
            tem_res = item + list_B[i]
            res.append(tem_res)

        return res

    @staticmethod
    def list_div(list_A: list, num: float) -> list:
        res = []

        for _, item in enumerate(list_A):
            tem_res = item / num
            res.append(tem_res)

        return res

    @staticmethod
    def list_mean(list_A: list) -> list:
        res = []

        for _, item in enumerate(list_A):
            res.append(item)

        return res

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
