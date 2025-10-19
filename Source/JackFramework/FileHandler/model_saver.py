# -*- coding: utf-8 -*-
"""Checkpoint persistence utilities."""

import importlib.util
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load module `{name}` from `{path}`.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


try:  # Prefer package-relative imports when available.
    from ..SysBasic import define as sys_def  # type: ignore
    from ..SysBasic.log_handler import LogHandler as log  # type: ignore
except ImportError:  # pragma: no cover - fallback when used outside package
    base_path = Path(__file__).resolve().parents[1]
    sys_def = _load_module_from_path('sys_define', base_path / 'SysBasic' / 'define.py')
    log_module = _load_module_from_path('log_handler', base_path / 'SysBasic' / 'log_handler.py')
    log = log_module.LogHandler

from .file_handler import FileHandler


class ModelSaver(object):
    """Manage serialisation of model/optimizer states and checkpoint metadata."""

    ROW_ONE = 1
    FIRST_SAVE_TIME = True
    FIST_SAVE_TIME = True  # Backward compatibility for legacy references.

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    @classmethod
    def _set_first_save_flag(cls, value: bool) -> None:
        cls.FIRST_SAVE_TIME = value
        cls.FIST_SAVE_TIME = value

    @staticmethod
    def __normalize_dir(path: str) -> Path:
        target = Path(path).expanduser()
        if target.is_file():
            raise ValueError(f'Expected directory path, received file: {target}')
        target.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def __checkpoint_list_path(root_dir: Path) -> Path:
        return root_dir / sys_def.CHECK_POINT_LIST_NAME

    # ------------------------------------------------------------------
    @staticmethod
    def __get_model_name(file_path: Path) -> str:
        header = FileHandler.get_line(str(file_path), ModelSaver.ROW_ONE)
        if not header.startswith(sys_def.LAST_MODEL_NAME):
            raise ValueError('Malformed checkpoint list header.')
        return header[len(sys_def.LAST_MODEL_NAME):]

    @classmethod
    def __consume_first_save(cls, list_path: Path) -> None:
        if cls.FIRST_SAVE_TIME:
            FileHandler.remove_file(str(list_path))
            cls._set_first_save_flag(False)

    @staticmethod
    def __read_existing_entries(list_path: Path) -> List[str]:
        if not list_path.exists():
            return []

        with list_path.open('r', encoding='utf-8') as handle:
            lines = [line.rstrip('\n') for line in handle]

        if not lines:
            return []

        header = lines[0]
        if not header.startswith(sys_def.LAST_MODEL_NAME):
            log.warning('The checkpoint list is malformed. Rewriting from scratch.')
            return []

        return lines

    @classmethod
    def __build_entry_list(cls, existing_lines: List[str], file_name: str) -> List[str]:
        new_entries = [f'{sys_def.LAST_MODEL_NAME}{file_name}']
        if existing_lines:
            # Skip the previous "latest" marker (index 0) but keep history.
            new_entries.extend(existing_lines[1:])
        new_entries.append(file_name)
        return new_entries

    @staticmethod
    def __write_entries(list_path: Path, entries: Iterable[str]) -> None:
        temp_path = list_path.with_suffix(list_path.suffix + '.tmp')
        with temp_path.open('w', encoding='utf-8') as handle:
            for entry in entries:
                handle.write(f'{entry}\n')
        temp_path.replace(list_path)

    # ------------------------------------------------------------------
    @classmethod
    def __write_checkpoint_metadata(cls, directory: Path, file_name: str) -> None:
        list_path = cls.__checkpoint_list_path(directory)
        cls.__consume_first_save(list_path)
        existing = cls.__read_existing_entries(list_path)
        entries = cls.__build_entry_list(existing, file_name)
        cls.__write_entries(list_path, entries)

    @staticmethod
    def __load_model_folder(path: Path) -> Optional[str]:
        log.info(f'Begin loading checkpoint from this folder: {path}')
        list_path = path / sys_def.CHECK_POINT_LIST_NAME
        if not list_path.is_file():
            log.warning(f"Checkpoint list file not found: {list_path}")
            return None

        try:
            checkpoint_name = ModelSaver.__get_model_name(list_path)
        except ValueError as exc:
            log.error(str(exc))
            return None

        checkpoint_path = path / checkpoint_name
        log.info(f'Get the path of model: {checkpoint_path}')
        return str(checkpoint_path)

    @staticmethod
    def __load_model_path(path: Path) -> str:
        log.info(f'Begin loading checkpoint from this file: {path}')
        return str(path)

    # Public API -------------------------------------------------------
    @staticmethod
    def get_check_point_path(path: str) -> Optional[str]:
        target = Path(path)
        if target.is_file():
            return ModelSaver.__load_model_path(target)
        return ModelSaver.__load_model_folder(target)

    @staticmethod
    def load_checkpoint(file_path: str, rank: Optional[int] = None) -> Dict:
        map_location = {'cuda:0': f'cuda:{rank}'} if rank is not None else None
        return torch.load(file_path, map_location=map_location)

    @staticmethod
    def load_model(model: object, checkpoint: Dict, model_id: int) -> None:
        key = f'model_{model_id}'
        if key not in checkpoint:
            raise KeyError(f'Model state `{key}` not found in checkpoint.')
        model.load_state_dict(checkpoint[key], strict=True)
        log.info('Model loaded successfully')

    @staticmethod
    def load_opt(opt: object, checkpoint: Dict, model_id: int) -> None:
        key = f'opt_{model_id}'
        if key not in checkpoint:
            raise KeyError(f'Optimizer state `{key}` not found in checkpoint.')
        opt.load_state_dict(checkpoint[key])
        log.info('Optimizer loaded successfully')

    @staticmethod
    def construct_model_dict(epoch: int, model_list: List[object], opt_list: List[object]) -> Dict[str, object]:
        if len(model_list) != len(opt_list):
            raise ValueError('Model and optimizer lists must be the same length.')

        model_dict: Dict[str, object] = {'epoch': epoch}
        for index, (model, opt) in enumerate(zip(model_list, opt_list)):
            model_dict[f'model_{index}'] = model.state_dict()
            model_dict[f'opt_{index}'] = opt.state_dict()
        return model_dict

    @classmethod
    def save(cls, file_dir: str, file_name: str, model_dict: Dict[str, object]) -> None:
        directory = cls.__normalize_dir(file_dir)
        checkpoint_path = directory / file_name
        torch.save(model_dict, checkpoint_path)
        log.info(f'Save model in : {checkpoint_path}')
        cls.__write_checkpoint_metadata(directory, file_name)

    @classmethod
    def write_check_point_list(cls, file_dir: str, file_name: str) -> None:
        directory = cls.__normalize_dir(file_dir)
        cls.__write_checkpoint_metadata(directory, file_name)
