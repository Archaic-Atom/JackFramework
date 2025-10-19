# -*- coding: utf-8 -*-
"""Formatting helpers for human-readable training logs."""

from typing import List, Optional, Sequence

DEFAULT_MAX_DECIMAL_PLACES = 6
DEFAULT_MIN_DECIMAL_PLACES = 2


class ResultStr(object):
    """Generate pre-formatted log strings for losses and metrics."""

    def __init__(self, arg=None) -> None:
        super().__init__()
        self.__arg = arg

    def training_result_str(self, epoch: int, loss: Sequence[float], acc: Sequence[float],
                            duration: Optional[float], training: bool = True) -> str:
        loss_str = self.loss2str(loss, decimal_places=DEFAULT_MAX_DECIMAL_PLACES)
        acc_str = self.acc2str(acc, decimal_places=DEFAULT_MAX_DECIMAL_PLACES)
        training_state = '[TrainProcess] ' if training else '[ValProcess] '
        duration_str = 'N/A s/epoch' if duration is None else f'{duration:.3f} s/epoch'
        return f"{training_state}e: {epoch}, {loss_str}, {acc_str} ({duration_str})"

    def testing_result_str(self, acc: Sequence[float], info_str: Sequence[str] = None) -> str:
        acc_str = self.acc2str(acc, info_str, decimal_places=DEFAULT_MAX_DECIMAL_PLACES)
        return f'[TestProcess] {acc_str}'

    def training_intermediate_result(self, epoch: int, loss: Sequence[float],
                                     acc: Sequence[float]) -> str:
        loss_str = self.loss2str(loss, decimal_places=3)
        acc_str = self.acc2str(acc, decimal_places=3)
        return f'e: {epoch}, {loss_str}, {acc_str}'

    def training_list_intermediate_result(self, epoch: int, loss: Sequence[Sequence[float]],
                                          acc: Sequence[Sequence[float]]) -> str:
        parts: List[str] = [f'e: {epoch}']
        for idx, (loss_item, acc_item) in enumerate(zip(loss, acc)):
            loss_str = self.loss2str(loss_item, decimal_places=3)
            acc_str = self.acc2str(acc_item, decimal_places=3)
            parts.append(f'model {idx}, {loss_str}, {acc_str}')
        return ', '.join(parts)

    def loss2str(self, loss: Sequence[float], info_str: Sequence[str] = None,
                 decimal_places: int = DEFAULT_MIN_DECIMAL_PLACES) -> str:
        labels = info_str or self.__gen_info_str('l', len(loss))
        return self.__data2str(loss, labels, decimal_places)

    def acc2str(self, acc: Sequence[float], info_str: Sequence[str] = None,
                decimal_places: int = DEFAULT_MIN_DECIMAL_PLACES) -> str:
        labels = info_str or self.__gen_info_str('a', len(acc))
        return self.__data2str(acc, labels, decimal_places)

    @staticmethod
    def __data2str(data: Sequence[float], labels: Sequence[str], decimal_places: int) -> str:
        if len(data) != len(labels):
            raise ValueError('Data and label lengths must match for formatting.')
        formatted = [f"{label}: {value:.{decimal_places}f}" for label, value in zip(labels, data)]
        return ', '.join(formatted)

    @staticmethod
    def __gen_info_str(prefix: str, num: int) -> List[str]:
        return [f'{prefix}{i}' for i in range(num)]
