# -*- coding: utf-8 -*-
"""Base accuracy helpers providing common configuration."""

from abc import ABCMeta


class MetaAccuracy(object, metaclass=ABCMeta):
    ACC_EPSILON = 1e-9

    @property
    def epsilon(self) -> float:
        return self.ACC_EPSILON
