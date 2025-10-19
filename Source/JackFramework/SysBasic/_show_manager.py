# -*- coding: utf-8 -*-
"""Helpers for gating display logic by distributed rank."""

from functools import wraps
from typing import Any, Callable, Optional


class ShowManager(object):
    """Coordinate UI rendering so only the default rank prints output."""

    __SHOW_MANAGER = None
    __DEFAULT_RANK_ID = 0
    __RANK: Optional[int] = None

    def __init__(self) -> None:
        super().__init__()

    @property
    def rank(self) -> Optional[int]:
        return self.__RANK

    @property
    def default_rank_id(self) -> int:
        return self.__DEFAULT_RANK_ID

    @staticmethod
    def get_rank() -> Optional[int]:
        return ShowManager.__RANK

    @staticmethod
    def set_rank(rank: Optional[int]) -> None:
        ShowManager.__RANK = rank

    @staticmethod
    def set_default_rank_id(default_rank_id: int) -> None:
        ShowManager.__DEFAULT_RANK_ID = default_rank_id

    @classmethod
    def show_method(cls, func: Callable[..., Any]) -> Callable[..., None]:
        @wraps(func)
        def wrapped_func(*args: Any, **kwargs: Any) -> None:
            if cls.__RANK == cls.__DEFAULT_RANK_ID or cls.__RANK is None:
                func(*args, **kwargs)

        return wrapped_func
