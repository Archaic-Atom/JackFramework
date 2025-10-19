# -*- coding: utf-8 -*-
"""Base class describing required data-loader hooks."""

from abc import ABCMeta, abstractmethod
from typing import List, Tuple


class DataHandlerTemplate(object, metaclass=ABCMeta):
    def __init__(self, args: object) -> None:
        super().__init__()
        self._args = args

    @abstractmethod
    def get_train_dataset(self, path: str, is_training: bool) -> object:
        """Return the dataset used for training or distributed sampling."""

    @abstractmethod
    def get_val_dataset(self, path: str) -> object:
        """Return the dataset used for evaluation/validation."""

    @abstractmethod
    def split_data(self, batch_data: Tuple, is_training: bool) -> List:
        """Split a dataloader batch into (inputs, labels/supplement)."""

    @abstractmethod
    def show_train_result(self, epoch: int, loss: List, acc: List, duration: float) -> None:
        """Pretty-print or log aggregated training results."""

    @abstractmethod
    def show_val_result(self, epoch: int, loss: List, acc: List, duration: float) -> None:
        """Pretty-print or log aggregated validation results."""

    @abstractmethod
    def save_result(self, output_data: List, supplement: List,
                    img_id: int, model_id: int) -> None:
        """Persist predictions for later inspection."""

    @abstractmethod
    def show_intermediate_result(self, epoch: int, loss: List, acc: List) -> str:
        """Return a one-line summary string for progress bars."""

    # Optional helpers for background/web modes -------------------------
    def load_test_data(self, cmd: str):  # pragma: no cover - optional extension point
        return None

    def save_test_data(self, output_data: List, supplement: List,
                       cmd: str, model_id: int):  # pragma: no cover
        return None
