# -*- coding: utf-8 -*-
"""Lightweight helper emulating a switch-case control flow."""

from typing import Any, Iterator


class Switch(object):
    def __init__(self, value: Any) -> None:
        self.__value = value
        self.__fall = False

    def __iter__(self) -> Iterator:
        """Return the match method once, then stop."""
        yield self.match
        return StopIteration

    def match(self, *args: Any) -> bool:
        """Indicate whether to enter a case suite."""
        if self.__fall or not args:
            return True
        if self.__value in args:
            self.__fall = True
            return True
        return False
