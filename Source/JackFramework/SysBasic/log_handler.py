# -*- coding: utf-8 -*-
"""Colored logging utilities for the framework."""

import logging
import os
from typing import Optional


class LogHandler(object):
    """Wrapper around Python logging that also prints colourised output."""

    LOG_FILE = 'output.log'
    LOG_FORMAT = '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    CONSOLE_FORMAT = '%(asctime)s │ %(levelname)-7s │ %(message)s'

    COLOR_SEQ_END = '\033[0m'
    LEVEL_COLORS = {
        'INFO': 32,
        'DEBUG': 36,
        'WARNING': 33,
        'ERROR': 31,
        'CRITICAL': 35,
    }

    LOGGER_NAME = 'JackFramework'

    def __init__(self, info_format: Optional[str] = None,
                 data_format: Optional[str] = None,
                 file_name: Optional[str] = None) -> None:
        super().__init__()
        self.__info_format = info_format or self.LOG_FORMAT
        self.__data_format = data_format or self.LOG_DATE_FORMAT
        self.__file_name = file_name or self.LOG_FILE

    def init_log(self, path: str, renew: bool) -> None:
        os.makedirs(path, exist_ok=True)
        log_path = os.path.join(path, self.__file_name)
        if renew and os.path.exists(log_path):
            os.remove(log_path)

        logger = logging.getLogger(self.LOGGER_NAME)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(self.__info_format, self.__data_format))
        logger.addHandler(file_handler)

        if self.__should_enable_console():
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._ConsoleFormatter(self.CONSOLE_FORMAT, self.__data_format))
            logger.addHandler(console_handler)

    def _disable_output_to_termimal(self) -> None:
        logger = logging.getLogger(self.LOGGER_NAME)
        logger.disabled = True

    def _eable_output_to_termimal(self) -> None:
        logger = logging.getLogger(self.LOGGER_NAME)
        logger.disabled = False

    @classmethod
    def _log(cls, level: str, message: str, *args: object, **kwargs: object) -> None:
        logger = logging.getLogger(cls.LOGGER_NAME)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format=cls.LOG_FORMAT,
                                datefmt=cls.LOG_DATE_FORMAT)
        getattr(logger, level)(message, *args, **kwargs)

    @classmethod
    def info(cls, data_str: str, *args: object, **kwargs: object) -> None:
        cls._log('info', data_str, *args, **kwargs)

    @classmethod
    def debug(cls, data_str: str, *args: object, **kwargs: object) -> None:
        cls._log('debug', data_str, *args, **kwargs)

    @classmethod
    def warning(cls, data_str: str, *args: object, **kwargs: object) -> None:
        cls._log('warning', data_str, *args, **kwargs)

    @classmethod
    def error(cls, data_str: str, *args: object, **kwargs: object) -> None:
        cls._log('error', data_str, *args, **kwargs)

    class _ConsoleFormatter(logging.Formatter):
        def __init__(self, fmt: str, datefmt: Optional[str]) -> None:
            super().__init__(fmt, datefmt)

        def format(self, record: logging.LogRecord) -> str:
            base = super().format(record)
            color_code = LogHandler.LEVEL_COLORS.get(record.levelname.upper())
            if color_code is None or not base:
                return base
            return f'\033[1;{color_code}m{base}{LogHandler.COLOR_SEQ_END}'

    @staticmethod
    def __should_enable_console() -> bool:
        # Debug override: show logs from all ranks when set
        if os.environ.get('JACK_LOG_ALL_RANKS', '0') == '1':
            return True
        rank_env = (os.environ.get('RANK') or
                    os.environ.get('LOCAL_RANK') or
                    os.environ.get('PROCESS_RANK'))
        if rank_env is None:
            return True
        try:
            return int(rank_env) == 0
        except ValueError:
            return False
