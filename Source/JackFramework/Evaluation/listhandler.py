# -*- coding: utf-8 -*-
"""Utility helpers for common list-based tensor aggregations."""

from typing import Iterable, List, Sequence


class ListHandler(object):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def list_add(list_a: Sequence[float], list_b: Sequence[float]) -> List[float]:
        if len(list_a) != len(list_b):
            raise ValueError('List lengths must match when adding.')
        return [a + b for a, b in zip(list_a, list_b)]

    @staticmethod
    def list_div(list_a: Sequence[float], num: float) -> List[float]:
        if num == 0:
            raise ZeroDivisionError('Cannot divide list values by zero.')
        return [item / num for item in list_a]

    @staticmethod
    def list_mean(list_a: Sequence[float]) -> List[float]:
        return list(list_a)

    @staticmethod
    def double_list_add(list_a: Sequence[Sequence[float]],
                        list_b: Sequence[Sequence[float]] = None) -> List[List[float]]:
        if not list_a:
            return []
        if list_b is None:
            return [ListHandler.list_mean(inner) for inner in list_a]
        if len(list_a) != len(list_b):
            raise ValueError('Outer list lengths must match when adding nested lists.')
        return [ListHandler.list_add(inner_a, inner_b)
                for inner_a, inner_b in zip(list_a, list_b)]

    @staticmethod
    def double_list_div(list_a: Sequence[Sequence[float]], num: float) -> List[List[float]]:
        return [ListHandler.list_div(inner, num) for inner in list_a]
