# -*- coding: utf-8 -*-
"""Minimal entrypoint to smoke test JackFramework end-to-end."""

from JackFramework import Application
from minimal_interface import MinimalInterface


if __name__ == '__main__':
    Application(MinimalInterface(), application_name='JF-Minimal').start()

