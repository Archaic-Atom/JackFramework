# -*- coding: utf-8 -*-
"""Lightweight helper emulating a switch-case control flow."""

from typing import Any, Iterator


class Switch(object):
    def __init__(self, value: Any) -> None:
        self.__value = value
        self.__fall = False

    def __iter__(self) -> Iterator:
        """Yield the match callable once, then stop iteration.

        产出一次 match 可调用对象后即停止迭代。
        """
        # A generator naturally raises StopIteration when exhausted; no
        # explicit return value is needed (and `return StopIteration` would
        # wrongly set the class object as the generator's return value).
        # 生成器耗尽时会自动抛出 StopIteration，无需显式返回值。
        yield self.match

    def match(self, *args: Any) -> bool:
        """Indicate whether to enter a case suite."""
        if self.__fall or not args:
            return True
        if self.__value in args:
            self.__fall = True
            return True
        return False
