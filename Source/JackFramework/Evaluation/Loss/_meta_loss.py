# -*- coding: utf-8 -*-
from abc import ABCMeta


class MetaLoss(object):
    __metaclass__ = ABCMeta
    __META_LOSS_INSTANCE = None
    LOSS_EPSILON = 1e-9

    def __new__(cls) -> object:
        if cls.__META_LOSS_INSTANCE is None:
            cls.__META_LOSS_INSTANCE = object.__new__(cls)
        return cls.__META_LOSS_INSTANCE

    @property
    def epsilon(self):
        return self.LOSS_EPSILON
