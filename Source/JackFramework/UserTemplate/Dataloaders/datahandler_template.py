# -*- coding: utf-8 -*-
"""Base class describing required data-loader hooks."""

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from JackFramework.UserTemplate._hook_validator import validate_hook_names


class DataHandlerTemplate(object, metaclass=ABCMeta):
    # Optional hooks JF dispatches via getattr; subclasses MAY override.
    # 框架可选 hook 名册，写错时 __init_subclass__ 报错。
    _OPTIONAL_HOOKS = frozenset({
        'load_test_data', 'save_test_data',
    })
    # Hard typo blacklist.
    # 硬黑名单。
    _HOOK_NAME_TYPOS = {
        'loadtestdata': 'load_test_data',
        'savetestdata': 'save_test_data',
        'load_testdata': 'load_test_data',
        'save_testdata': 'save_test_data',
    }

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        validate_hook_names(
            cls,
            optional_hooks=cls._OPTIONAL_HOOKS,
            typo_map=cls._HOOK_NAME_TYPOS,
            template_name='DataHandlerTemplate',
        )

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
