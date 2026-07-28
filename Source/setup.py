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


# Packages that ``import JackFramework`` pulls in EAGERLY. Determined by
# diffing ``sys.modules`` across the import in a clean environment, not by
# reading the source — a lazily imported module must not be listed here,
# and an eagerly imported one that is missing makes the package unusable
# the moment it is installed.
#
# ``import JackFramework`` 会**急切加载**的包。通过在干净环境中对比导入
# 前后的 ``sys.modules`` 得出，而不是靠读源码 —— 懒加载的模块不该列在
# 这里，而急切加载的模块一旦缺失，装完就直接不可用。
#
# Notes / 说明:
#   * ``torch`` is intentionally unpinned: users pick the build that matches
#     their CUDA. An already-installed torch satisfies this requirement, so
#     declaring it will not disturb an existing conda environment.
#     ``torch`` 故意不锁版本：用户按自己的 CUDA 选择构建版本。环境里已有的
#     torch 就能满足该依赖，因此声明它不会打扰现有 conda 环境。
#   * ``opencv-python`` rather than ``opencv-python-headless``: the framework
#     only calls imread/resize and would work with either, but most users
#     already have the regular build, and requiring the headless one would
#     install a SECOND cv2 alongside it. Headless is a fine manual swap on
#     slim containers that lack libGL.
#     选 ``opencv-python`` 而非 ``opencv-python-headless``：框架只用到
#     imread/resize，两者都可以；但多数用户环境里已经装了常规版，若要求
#     headless 会再装一个 cv2 并存。缺少 libGL 的精简容器里可手动换成
#     headless。
#   * ``django`` is a hard requirement only because ``Core/Mode`` imports the
#     web mode eagerly. Making that import lazy would let django move to an
#     optional extra — worth doing, but it is a behaviour change, not a
#     packaging one.
#     ``django`` 成为硬依赖，仅仅是因为 ``Core/Mode`` 急切导入了 web 模式。
#     把该导入改成懒加载就能把 django 降级为可选附加项 —— 值得做，但那是
#     行为变更而非打包变更。
INSTALL_REQUIRES = [
    'torch',
    'numpy',
    'opencv-python',
    'pillow',
    'django',
    'tensorboard',      # required by torch.utils.tensorboard
]


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
        install_requires=INSTALL_REQUIRES,
    )


if __name__ == '__main__':
    main()
