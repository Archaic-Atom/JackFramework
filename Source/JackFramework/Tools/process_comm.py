# -*- coding: utf-8 -*-
"""Utilities for inter-process communication based on named pipes."""

import os
import threading
import time
from queue import Empty, Full, Queue
from typing import Optional, Tuple

import JackFramework.SysBasic.define as sys_def
from JackFramework.SysBasic.log_handler import LogHandler as log


class NamedPipe(object):
    """Simple singleton wrapper around a pair of named pipes."""

    __SERVER_OBJECT = None
    __EXIT_WAIT_TIME = 5
    __THREAD_WAIT_TIME = 1
    __BUFF_SIZE = 1024

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__SERVER_OBJECT is None:
            cls.__SERVER_OBJECT = object.__new__(cls)
        return cls.__SERVER_OBJECT

    def __init__(self, mode: str = 'server', queue_size: int = 10000,
                 writer_path: Optional[str] = None, reader_path: Optional[str] = None) -> None:
        super().__init__()
        if mode not in {'server', 'client'}:
            raise ValueError("mode must be either 'server' or 'client'.")

        # Re-initialise cleanly if the singleton is reused with new settings.
        if getattr(self, '_NamedPipe__initialised', False):
            self.close()

        self.__mode = mode
        self.__msg_queue = Queue(maxsize=queue_size)
        self.__exit_event = threading.Event()
        self.__pipe_writer: Optional[int] = None
        self.__pipe_reader: Optional[int] = None
        self.__receive_thread: Optional[threading.Thread] = None
        self.__closed = False

        self.__pipe_writer, self.__pipe_reader = self.__create_pipe(writer_path, reader_path)
        self.__receive_thread = threading.Thread(
            target=self.__receive_loop,
            name=f'NamedPipe-{self.__mode}-receiver',
            args=(reader_path,),
            daemon=True,
        )
        self.__receive_thread.start()

        self.__initialised = True

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Gracefully stop the background thread and release descriptors."""

        if getattr(self, '_NamedPipe__closed', False):
            return

        self.__exit_event.set()
        # Closing the descriptors wakes the blocking read immediately.
        self.__close_pipe()

        if self.__receive_thread is not None and self.__receive_thread.is_alive():
            self.__receive_thread.join(timeout=self.__EXIT_WAIT_TIME)
            if self.__receive_thread.is_alive():
                log.warning(f'Receiver thread for {self.__mode} did not exit within timeout.')

        self.__closed = True
        self.__initialised = False
        log.info(f'The receive thread in {self.__mode} has been stopped.')

    @staticmethod
    def __get_pipe_default_path(mode: str = 'server') -> Tuple[str, str]:
        if mode == 'server':
            return sys_def.PIPE_READ_PATH, sys_def.PIPE_WRITE_PATH
        return sys_def.PIPE_WRITE_PATH, sys_def.PIPE_READ_PATH

    def __close_pipe(self) -> None:
        if self.__pipe_writer is not None:
            os.close(self.__pipe_writer)
            self.__pipe_writer = None
        if self.__pipe_reader is not None:
            os.close(self.__pipe_reader)
            self.__pipe_reader = None

    def __get_path(self, writer_path: Optional[str] = None,
                   reader_path: Optional[str] = None) -> Tuple[str, str]:
        if writer_path is None:
            writer_path, _ = self.__get_pipe_default_path(self.__mode)
        if reader_path is None:
            _, reader_path = self.__get_pipe_default_path(self.__mode)
        return writer_path, reader_path

    @staticmethod
    def __create_pipe_file(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
        try:
            os.mkfifo(path)
        except OSError as exc:
            log.error(f'Failed to create pipe file at {path}: {exc}')
            raise

    def __create_file(self, writer_path: str, reader_path: str) -> None:
        if self.__mode == 'server':
            self.__create_pipe_file(writer_path)
            self.__create_pipe_file(reader_path)

    def __create_sender_pipe(self, writer_path: str) -> Optional[int]:
        if not os.path.exists(writer_path):
            log.warning(f'Sender path {writer_path} does not exist for {self.__mode}.')
            return None

        log.info(f'{self.__mode} creates a sender at {writer_path}')
        return os.open(writer_path, os.O_SYNC | os.O_CREAT | os.O_RDWR)

    def __create_receiver_pipe(self, reader_path: str) -> Optional[int]:
        if not os.path.exists(reader_path):
            log.warning(f'Receiver path {reader_path} does not exist for {self.__mode}.')
            return None

        log.info(f'{self.__mode} creates a receiver at {reader_path}')
        log.info(f'{self.__mode} is waiting for messages')
        return os.open(reader_path, os.O_RDONLY)

    def __create_pipe(self, writer_path: Optional[str],
                      reader_path: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
        writer_path, reader_path = self.__get_path(writer_path, reader_path)
        self.__create_file(writer_path, reader_path)
        log.info(f"Pipe paths for {self.__mode}: writer={writer_path}, reader={reader_path}")

        pipe_writer = self.__create_sender_pipe(writer_path)
        pipe_reader = self.__create_receiver_pipe(reader_path) if self.__mode == 'server' else None
        return pipe_writer, pipe_reader

    def send(self, msg: str) -> None:
        """Send a message through the named pipe."""

        if self.__pipe_writer is None:
            raise RuntimeError('Named pipe writer has not been initialised.')

        os.write(self.__pipe_writer, msg.encode())

    def receive(self, timeout: Optional[float] = None) -> str:
        """Retrieve a message from the receive queue."""

        try:
            return self.__msg_queue.get(timeout=timeout)
        except Empty as exc:
            raise TimeoutError('Timed out waiting for named pipe message.') from exc

    def __receive_loop(self, reader_path_hint: Optional[str]) -> None:
        log.info('The receive thread starts!')
        while not self.__exit_event.is_set():
            if self.__pipe_reader is None:
                _, reader_path = self.__get_path(None, reader_path_hint)
                self.__pipe_reader = self.__create_receiver_pipe(reader_path)
                if self.__pipe_reader is None:
                    time.sleep(self.__THREAD_WAIT_TIME)
                    continue

            try:
                msg = os.read(self.__pipe_reader, self.__BUFF_SIZE)
            except OSError as exc:
                log.warning(f'Error reading from pipe: {exc}')
                time.sleep(self.__THREAD_WAIT_TIME)
                continue

            if not msg:
                time.sleep(self.__THREAD_WAIT_TIME)
                continue

            try:
                self.__msg_queue.put(msg.decode(), timeout=self.__THREAD_WAIT_TIME)
            except Full:
                log.warning('Named pipe queue is full; dropping message.')

        log.info(f'The receive thread in {self.__mode} has exited!')
