# -*- coding: utf-8 -*-
from functools import wraps


class ShowManager(object):
    __SHOW_MANAGER = None
    __DEFAULT_RANK_ID = 0
    __RANK = None

    def __init__(self) -> None:
        super().__init__()

    @property
    def rank(self) -> object:
        return self.__RANK

    @property
    def default_rank_id(self):
        return self.__DEFAULT_RANK_ID

    @staticmethod
    def get_rank() -> object:
        return ShowManager.__RANK

    @staticmethod
    def set_rank(rank: object) -> None:
        ShowManager.__RANK = rank

    @staticmethod
    def set_default_rank_id(default_rank_id: int) -> None:
        ShowManager.__DEFAULT_RANK_ID = default_rank_id

    @classmethod
    def show_method(cls, func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if cls.__RANK == cls.__DEFAULT_RANK_ID or cls.__RANK is None:
                func(*args, **kwargs)
        return wrapped_func
