# -*- coding: utf-8 -*-
"""Lightweight filesystem helpers used across the framework."""

import linecache
import os
from pathlib import Path
from typing import IO, List, Optional


class FileHandler(object):
    """Utility wrapper around common file-system operations."""

    ERROR_LINE_NUM = -1

    def __init__(self) -> None:
        super().__init__()

    # Directory helpers -------------------------------------------------
    @staticmethod
    def mkdir(path: str) -> None:
        """Create *path* (and parents) if it does not already exist."""

        target = Path(path).expanduser()
        if target.is_file():
            raise ValueError(f'Unable to create directory because a file already exists at {target}')
        target.mkdir(parents=True, exist_ok=True)

    # File helpers ------------------------------------------------------
    @staticmethod
    def open_file(path: str, is_continue: bool = True, mode: str = 'a+', encoding: str = 'utf-8') -> IO[str]:
        """Open *path* with sensible defaults and optional truncation."""

        target = Path(path).expanduser()
        if not is_continue and target.exists():
            target.unlink()

        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)

        # Ensure text mode includes reading for random access.
        if 'b' in mode:
            return target.open(mode)
        return target.open(mode, encoding=encoding)

    @staticmethod
    def remove_file(path: str) -> None:
        target = Path(path)
        if target.is_file():
            target.unlink()

    @staticmethod
    def close_file(fd_file: Optional[IO[str]]) -> None:
        if fd_file is not None:
            fd_file.close()

    @staticmethod
    def write_file(fd_file: IO[str], data_str: str) -> None:
        fd_file.write(f'{data_str}\n')
        fd_file.flush()

    # Reading helpers ---------------------------------------------------
    @staticmethod
    def get_line(filename: str, line_num: int) -> str:
        line = linecache.getline(filename, line_num)
        return line.rstrip('\n')

    @staticmethod
    def insert_str2line(fd_file: IO[str], data_str: str, line_num: int) -> None:
        off_set = FileHandler.__get_line_offset(fd_file, line_num)
        if off_set == FileHandler.ERROR_LINE_NUM:
            raise IndexError('Line number out of range when inserting text.')
        fd_file.seek(off_set)
        FileHandler.write_file(fd_file, data_str)

    @staticmethod
    def get_line_fd(fd_file: IO[str], line_num: int) -> str:
        current_off_set = fd_file.tell()
        fd_file.seek(0, os.SEEK_SET)
        offsets = FileHandler.__line2offset(fd_file)
        if line_num >= len(offsets):
            fd_file.seek(current_off_set, os.SEEK_SET)
            return ''
        fd_file.seek(offsets[line_num], os.SEEK_SET)
        line = fd_file.readline().rstrip('\n')
        fd_file.seek(current_off_set, os.SEEK_SET)
        return line

    @staticmethod
    def copy_file(fd_file_a: IO[str], fd_file_b: IO[str], line_num: int) -> None:
        source_offset = FileHandler.__get_line_offset(fd_file_a, line_num)
        if source_offset == FileHandler.ERROR_LINE_NUM:
            return
        fd_file_a.seek(source_offset, os.SEEK_SET)

        dest_offset = FileHandler.__get_line_offset(fd_file_b, line_num)
        if dest_offset != FileHandler.ERROR_LINE_NUM:
            fd_file_b.seek(dest_offset, os.SEEK_SET)

        for next_line in fd_file_a:
            FileHandler.write_file(fd_file_b, next_line.rstrip('\n'))

    # Internal helpers --------------------------------------------------
    @staticmethod
    def __get_line_offset(fd_file: IO[str], line_num: int) -> int:
        offsets = FileHandler.__line2offset(fd_file)
        if line_num >= len(offsets):
            return FileHandler.ERROR_LINE_NUM
        return offsets[line_num]

    @staticmethod
    def __line2offset(fd_file: IO[str]) -> List[int]:
        current_off_set = fd_file.tell()
        fd_file.seek(0, os.SEEK_SET)
        offsets: List[int] = [0]
        running_total = 0

        for line in fd_file:
            running_total += len(line)
            offsets.append(running_total)

        fd_file.seek(current_off_set, os.SEEK_SET)
        return offsets
