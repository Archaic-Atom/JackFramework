# -*- coding: utf-8 -*-
"""Base class describing the hooks a user model must implement."""

from abc import ABCMeta, abstractmethod
from typing import List, Sequence, Tuple

from JackFramework.UserTemplate._hook_validator import validate_hook_names


class ModelHandlerTemplate(object, metaclass=ABCMeta):
    # Optional hooks JF dispatches via getattr; subclasses MAY override
    # any subset. Listed here so ``__init_subclass__`` can catch typos
    # at class-definition time (e.g. ``postprocess`` vs ``post_process``).
    # 框架可选 hook 名册，写错时 __init_subclass__ 报错。
    _OPTIONAL_HOOKS = frozenset({
        'pretreatment', 'post_process',
        'load_model', 'load_opt', 'save_model',
    })
    # Hard typo blacklist (names we've actually seen in real projects).
    # 硬黑名单：踩过的拼写错。
    _HOOK_NAME_TYPOS = {
        'postprocess':    'post_process',
        'postProcess':    'post_process',
        'preprocess':     'pretreatment',
        'pretrain':       'pretreatment',
        'loadmodel':      'load_model',
        'savemodel':      'save_model',
        'load_optimizer': 'load_opt',
    }

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        validate_hook_names(
            cls,
            optional_hooks=cls._OPTIONAL_HOOKS,
            typo_map=cls._HOOK_NAME_TYPOS,
            template_name='ModelHandlerTemplate',
        )

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
