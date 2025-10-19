# -*- coding: utf-8 -*-
"""Colored logging utilities for the framework."""

import logging
import os


class LogHandler(object):
    """Wrapper around Python logging that also prints colourised output."""

    LOG_FILE = 'output.log'
    LOG_FORMAT = '[%(levelname)s] %(asctime)s %(filename)s[line:%(lineno)d]: %(message)s'
    LOG_DATE_FORMAT = '[%a] %Y-%m-%d %H:%M:%S'

    COLOR_SEQ_HEAD = '\033[1;%dm'
    COLOR_SEQ_END = '\033[0m'

    COLOR_GREEN = 32
    COLOR_YELLOW = 33
    COLOR_RED = 31

    LOGGER_NAME = 'JackFramework'

    def __init__(self, info_format: str = None, data_format: str = None,
                 file_name: str = None) -> None:
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

    def _disable_output_to_termimal(self) -> None:
        logger = logging.getLogger(self.LOGGER_NAME)
        logger.disabled = True

    def _eable_output_to_termimal(self) -> None:
        logger = logging.getLogger(self.LOGGER_NAME)
        logger.disabled = False

    @classmethod
    def _log(cls, level: str, color: int, message: str) -> None:
        logger = logging.getLogger(cls.LOGGER_NAME)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format=cls.LOG_FORMAT,
                                datefmt=cls.LOG_DATE_FORMAT)
        print(cls.COLOR_SEQ_HEAD % color + f'[{level.upper()}] ' + message + cls.COLOR_SEQ_END)
        getattr(logger, level)(message)

    @classmethod
    def info(cls, data_str: str) -> None:
        cls._log('info', cls.COLOR_GREEN, data_str)

    @classmethod
    def debug(cls, data_str: str) -> None:
        cls._log('debug', cls.COLOR_GREEN, data_str)

    @classmethod
    def warning(cls, data_str: str) -> None:
        cls._log('warning', cls.COLOR_YELLOW, data_str)

    @classmethod
    def error(cls, data_str: str) -> None:
        cls._log('error', cls.COLOR_RED, data_str)
