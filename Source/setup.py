# -*- coding: utf-8 -*-
import os
from setuptools import find_packages, setup


def get_all_pkg(lib_name: str) -> list:
    all_sub_pkg = find_packages('JackFramework')
    all_pkg = [lib_name + '.' + i for i in all_sub_pkg]
    all_pkg.insert(0, lib_name)
    return all_pkg


def install_lib(lib_name: str, all_pkg: list,
                version: str, description: str,
                author: str, lic: str) -> None:
    setup(name=lib_name, packages=all_pkg, version=version,
          description=description, author=author, license=lic,
          )


def main():
    lib_name = 'JackFramework'
    version = '0.1.1'
    description = 'The deep learning tranining framework based on pytorch.'
    author = 'Jack Rao'
    lic = 'MIT'
    all_pkg = get_all_pkg(lib_name)
    install_lib(lib_name, all_pkg, version, description, author, lic)


if __name__ == '__main__':
    main()
