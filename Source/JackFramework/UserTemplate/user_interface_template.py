# -*- coding: utf-8 -*-
"""Template class that user-facing interfaces should extend."""

import argparse
from abc import ABCMeta, abstractmethod
from typing import Tuple


class NetWorkInferenceTemplate(object, metaclass=ABCMeta):
    __NETWORK_INFERENCE = None

    def __new__(cls, *args: object, **kwargs: object) -> 'NetWorkInferenceTemplate':
        if cls.__NETWORK_INFERENCE is None:
            cls.__NETWORK_INFERENCE = super().__new__(cls)
        return cls.__NETWORK_INFERENCE

    @abstractmethod
    def inference(self, args: object) -> Tuple[object, object]:
        """Return instantiated (ModelHandlerTemplate, DataHandlerTemplate)."""

    @abstractmethod
    def user_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Optionally extend the default CLI parser with user arguments."""

    @staticmethod
    def _str2bool(arg: str) -> bool:
        lowered = arg.lower()
        if lowered in {'yes', 'true', 't', 'y', '1'}:
            return True
        if lowered in {'no', 'false', 'f', 'n', '0'}:
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')
