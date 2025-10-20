# -*- coding: utf-8 -*-
"""Minimal, dependency-free user interface for smoke testing JackFramework."""

from __future__ import annotations

import argparse
from typing import List, Sequence, Tuple

import torch
from torch import nn

from JackFramework.UserTemplate.user_interface_template import NetWorkInferenceTemplate
from JackFramework.UserTemplate.Models.ModelTemplate.model_template import (
    ModelHandlerTemplate,
)
from JackFramework.UserTemplate.Dataloaders.datahandler_template import (
    DataHandlerTemplate,
)
from JackFramework.SysBasic.log_handler import LogHandler as log


class _RandomDataset(torch.utils.data.Dataset):
    def __init__(self, length: int, input_shape: Tuple[int, int, int], label_dim: int) -> None:
        super().__init__()
        self.length = max(int(length), 0)
        self.input_shape = input_shape
        self.label_dim = label_dim

    def __len__(self) -> int:  # type: ignore[override]
        return self.length

    def __getitem__(self, idx: int):  # type: ignore[override]
        x = torch.randn(self.input_shape, dtype=torch.float32)
        y = torch.randn(self.label_dim, dtype=torch.float32)
        return x, y


class _MinimalModel(ModelHandlerTemplate):
    def __init__(self, args: object, input_shape: Tuple[int, int, int], label_dim: int) -> None:
        super().__init__(args)
        c, h, w = input_shape
        in_features = c * h * w
        self._net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, label_dim),
        )

    def get_model(self) -> Sequence[object]:
        return [self._net]

    def inference(self, model: nn.Module, input_data: List, model_id: int) -> List:
        x = input_data[0]
        return [model(x)]

    def optimizer(self, model: Sequence[object], lr: float):
        params = list(model[0].parameters())
        opt = torch.optim.SGD(params, lr=lr, momentum=0.0)
        return [opt], [None]

    def lr_scheduler(self, sch: object, ave_loss: float, sch_id: int) -> None:
        return None

    def accuracy(self, output_data: List, label_data: List, model_id: int) -> List[float]:
        # 1 - mean absolute error (for demo only)
        out, lab = output_data[0], label_data[0]
        return [1.0 - torch.mean(torch.abs(out - lab))]

    def loss(self, output_data: List, label_data: List, model_id: int) -> List:
        out, lab = output_data[0], label_data[0]
        return [torch.nn.functional.mse_loss(out, lab)]


class _MinimalData(DataHandlerTemplate):
    def __init__(self, args: object, input_shape: Tuple[int, int, int], label_dim: int) -> None:
        super().__init__(args)
        self._shape = input_shape
        self._label_dim = label_dim

    def get_train_dataset(self, path: str, is_training: bool) -> object:
        return _RandomDataset(self._args.imgNum, self._shape, self._label_dim)

    def get_val_dataset(self, path: str) -> object:
        return _RandomDataset(self._args.valImgNum, self._shape, self._label_dim)

    def split_data(self, batch_data: Tuple, is_training: bool) -> List:
        x, y = batch_data
        return [x], [y]

    def show_train_result(self, epoch: int, loss: List, acc: List, duration: float) -> None:
        l0 = float(loss[0][0]) if loss and loss[0] else 0.0
        a0 = float(acc[0][0]) if acc and acc[0] else 0.0
        log.info(f"[TrainProcess] e: {epoch}, l0: {l0:.3f}, a0: {a0:.3f} ({duration if duration is not None else 0:.3f} s/epoch)")

    def show_val_result(self, epoch: int, loss: List, acc: List, duration: float) -> None:
        l0 = float(loss[0][0]) if loss and loss[0] else 0.0
        a0 = float(acc[0][0]) if acc and acc[0] else 0.0
        log.info(f"[ValProcess] e: {epoch}, l0: {l0:.3f}, a0: {a0:.3f} ({duration if duration is not None else 0:.3f} s/epoch)")

    def save_result(self, output_data: List, supplement: List, img_id: int, model_id: int) -> None:
        return None

    def show_intermediate_result(self, epoch: int, loss: List, acc: List) -> str:
        try:
            l0 = float(loss[0][0]) if loss and loss[0] else 0.0
            a0 = float(acc[0][0]) if acc and acc[0] else 0.0
        except Exception:
            l0, a0 = 0.0, 0.0
        return f"e: {epoch}, l0: {l0:.3f}, a0: {a0:.3f}"


class MinimalInterface(NetWorkInferenceTemplate):
    """A tiny end-to-end setup producing random data and training a MLP."""

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 32, 32),
                 label_dim: int = 10) -> None:
        super().__init__()
        self._input_shape = input_shape
        self._label_dim = label_dim

    def inference(self, args: object) -> Tuple[object, object]:
        model = _MinimalModel(args, self._input_shape, self._label_dim)
        data = _MinimalData(args, self._input_shape, self._label_dim)
        return model, data

    def user_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Provide a convenient minimal preset for quick runs
        parser.add_argument('--mini_preset', default=True, type=self._str2bool,
                            help='Use a tiny random dataset and short run (default: true)')
        return parser

