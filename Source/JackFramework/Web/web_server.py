#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from pathlib import Path

import django
try:
    from django.core.management import call_command, execute_from_command_line
except ImportError as exc:
    raise ImportError(
        "Couldn't import Django. Are you sure it's installed and "
        "available on your PYTHONPATH environment variable? Did you "
        "forget to activate a virtual environment?"
    ) from exc

from JackFramework.SysBasic.log_handler import LogHandler as log


class WebServer(object):
    """docstring for ClassName"""

    def __init__(self, args: object):
        super().__init__()
        self.__args = args

    def set_env(self) -> None:
        log.info(f'the web server path: {str(Path(__file__).resolve().parent)}')
        sys.path.append(str(Path(__file__).resolve().parent))
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'jackframework_webserver.settings')

    def start_web(self) -> None:
        args = self.__args
        django.setup()
        call_command('makemigrations')
        call_command('migrate')
        log.info(f'the web cmd: {args.web_cmd}')
        execute_from_command_line(args.web_cmd.split(' '))

    def exec(self) -> None:
        self.set_env()
        self.start_web()
