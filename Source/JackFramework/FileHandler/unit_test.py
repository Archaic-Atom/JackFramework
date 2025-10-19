# -*- coding: utf-8 -*-
"""Smoke tests for the file handler and checkpoint utilities."""

import importlib.util
import tempfile
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent


def _load_module(qualname: str, path: Path):
    spec = importlib.util.spec_from_file_location(qualname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load module `{qualname}` from `{path}`.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


FileHandler = _load_module('JackFramework.FileHandler.file_handler', MODULE_DIR / 'file_handler.py').FileHandler
ModelSaver = _load_module('JackFramework.FileHandler.model_saver', MODULE_DIR / 'model_saver.py').ModelSaver


class FileHandlerUnitTestFramework(object):
    CHECK_POINT_LIST_NAME = 'checkpoint.list'

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _test_model_saver() -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ModelSaver.write_check_point_list(tmp_dir, 'test_model_epoch_1.pth')
            ModelSaver.write_check_point_list(tmp_dir, 'test_model_epoch_2.pth')

            list_path = Path(tmp_dir) / FileHandlerUnitTestFramework.CHECK_POINT_LIST_NAME
            print(list_path.read_text().splitlines())

    @staticmethod
    def _test_file_handler() -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)
            source_path = base_path / 'source.txt'
            dest_path = base_path / 'dest.txt'

            source = FileHandler.open_file(str(source_path), is_continue=False)
            FileHandler.write_file(source, 'header')
            FileHandler.write_file(source, 'line_a')
            FileHandler.write_file(source, 'line_b')
            FileHandler.close_file(source)

            source = FileHandler.open_file(str(source_path))
            dest = FileHandler.open_file(str(dest_path), is_continue=False)
            FileHandler.write_file(dest, 'header')
            FileHandler.copy_file(source, dest, 1)
            FileHandler.close_file(source)
            FileHandler.close_file(dest)

            print(dest_path.read_text().splitlines())

    def test(self) -> None:
        self._test_model_saver()
        self._test_file_handler()


def main() -> None:
    test_framework = FileHandlerUnitTestFramework()
    test_framework.test()


if __name__ == '__main__':
    main()
