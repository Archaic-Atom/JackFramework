# -*- coding: utf-8 -*-
"""Base class describing the hooks a user model must implement."""

from abc import ABCMeta, abstractmethod
from typing import List, Sequence, Tuple


class ModelHandlerTemplate(object, metaclass=ABCMeta):
    def __init__(self, args: object) -> None:
        super().__init__()
        self._args = args

    @abstractmethod
    def get_model(self) -> Sequence[object]:
        """Build and return the sequence of model replicas."""

    @abstractmethod
    def inference(self, model: object, input_data: List, model_id: int) -> List:
        """Run forward inference for a given model replica."""

    @abstractmethod
    def optimizer(self, model: Sequence[object], lr: float) -> Tuple[Sequence[object], Sequence[object]]:
        """Create optimizer(s) and optional scheduler(s) matching the model list."""

    @abstractmethod
    def lr_scheduler(self, sch: object, ave_loss: float, sch_id: int) -> None:
        """Advance the learning rate scheduler for a given model index."""

    @abstractmethod
    def accuracy(self, output_data: List, label_data: List, model_id: int) -> List[float]:
        """Compute accuracy metrics for a single replica."""

    @abstractmethod
    def loss(self, output_data: List, label_data: List, model_id: int) -> List:
        """Compute loss tensors for backpropagation."""

    def pretreatment(self, epoch: int, rank: object) -> None:
        """Optional hook executed before each training epoch."""

    def post_process(self, epoch: int, rank: object,
                     ave_tower_loss: List, ave_tower_acc: List) -> None:
        """Optional hook executed after each epoch."""

    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        """Optional custom model load that can override framework defaults."""
        return False

    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        """Optional custom optimizer load that can override framework defaults."""
        return False

    def save_model(self, epoch: int, model_list: Sequence[object],
                   opt_list: Sequence[object]) -> dict:
        """Optional serialisation override; return None to use defaults."""
        return None
