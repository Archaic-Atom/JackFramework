#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setuptools configuration for packaging JackFramework."""

from pathlib import Path
from typing import Dict

from setuptools import find_packages, setup

PACKAGE_NAME = 'JackFramework'
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
README_PATH = ROOT_DIR / 'README.md'
VERSION_MODULE = BASE_DIR / PACKAGE_NAME / 'SysBasic' / 'define.py'


def read_version() -> str:
    namespace: Dict[str, str] = {}
    with VERSION_MODULE.open('r', encoding='utf-8') as handle:
        exec(handle.read(), namespace)
    version = namespace.get('VERSION')
    if not version:
        raise RuntimeError('VERSION is not defined in define.py')
    return version


def read_readme() -> str:
    if not README_PATH.exists():
        return ''
    return README_PATH.read_text(encoding='utf-8')


def main() -> None:
    setup(
        name=PACKAGE_NAME,
        version=read_version(),
        description='The deep learning training framework based on PyTorch.',
        long_description=read_readme(),
        long_description_content_type='text/markdown',
        author='Jack Rao',
        license='MIT',
        packages=find_packages(include=[PACKAGE_NAME, f'{PACKAGE_NAME}.*']),
        include_package_data=True,
        python_requires='>=3.8',
    )


if __name__ == '__main__':
    main()
