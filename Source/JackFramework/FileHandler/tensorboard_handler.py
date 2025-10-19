# -*- coding: utf-8 -*-
"""Wrapper around `SummaryWriter` with consistent naming conventions."""

from typing import Iterable, Optional, Sequence

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore


class TensorboardHandler(object):
    """Singleton interface for writing scalar metrics to TensorBoard."""

    __TENSORBOARD_HANDLER = None

    def __new__(cls, *args: object, **kwargs: object) -> object:
        if cls.__TENSORBOARD_HANDLER is None:
            cls.__TENSORBOARD_HANDLER = object.__new__(cls)
        return cls.__TENSORBOARD_HANDLER

    def __init__(self, args: object) -> None:
        super().__init__()
        if SummaryWriter is None:
            raise RuntimeError('TensorBoard support requires `tensorboard` to be installed.')

        self.__args = args
        self.__writer: Optional[SummaryWriter] = SummaryWriter(log_dir=args.log)

    def close(self) -> None:
        if self.__writer is not None:
            self.__writer.flush()
            self.__writer.close()
            self.__writer = None

    def __del__(self) -> None:
        self.close()

    def _write_scalar_group(self, epoch: int, model_id: int,
                             template: str, values: Iterable[float],
                             data_state: str) -> None:
        if self.__writer is None:
            return

        for metric_id, value in enumerate(values):
            tag = template % (model_id, metric_id, data_state)
            self.__writer.add_scalar(tag, value, epoch)

    def write_data(self, epoch: int, model_loss_list: Sequence[Sequence[float]],
                   model_acc_list: Sequence[Sequence[float]], data_state: str) -> None:
        if len(model_loss_list) != len(model_acc_list):
            raise ValueError('Loss and accuracy collections must have identical length.')

        data_loss_title = 'model:%d/l%d/%s'
        data_acc_title = 'model:%d/acc%d/%s'

        for model_id, loss_list_item in enumerate(model_loss_list):
            self._write_scalar_group(epoch, model_id, data_loss_title, loss_list_item, data_state)
            self._write_scalar_group(epoch, model_id, data_acc_title, model_acc_list[model_id], data_state)
