# -*- coding: utf-8 -*-
from abc import ABCMeta


class MetaAccuracy(object):
    __metaclass__ = ABCMeta
    __META_ACCURACY_INSTANCE = None
    ACC_EPSILON = 1e-9

    def __new__(cls) -> object:
        if cls.__META_ACCURACY_INSTANCE is None:
            cls.__META_ACCURACY_INSTANCE = object.__new__(cls)
        return cls.__META_ACCURACY_INSTANCE

    @property
    def epsilon(self):
        return self.ACC_EPSILON
