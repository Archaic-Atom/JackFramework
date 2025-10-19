# -*- coding: utf-8 -*-
"""Base loss helpers providing shared epsilon configuration."""

from abc import ABCMeta


class MetaLoss(object, metaclass=ABCMeta):
    LOSS_EPSILON = 1e-9

    @property
    def epsilon(self) -> float:
        return self.LOSS_EPSILON
