# -*- coding: utf-8 -*-
import os
import time
import queue as Queue
import _thread as thread

import JackFramework.SysBasic.define as sys_def
from JackFramework.SysBasic.log_handler import LogHandler as log


class NamedPipe(object):
    """docstring for ClassName"""
    __SERVER_OBJECT = None
    __EXIT_WAIT_TIME = 5
    __THREAD_WAIT_TIME = 1
    __BUFF_SIZE = 1024

    def __init__(self, mode: str = 'server', queue_size: int = 10000,
                 writer_path: str = None, reader_path: str = None) -> None:
        super().__init__()
        assert mode in {'server', 'client'}
        self.__mode = mode
        self.__msg_queue = Queue.Queue(maxsize=queue_size)
        self.__exit = False
        self.__pipe_writer, self.__pipe_reader = self.__create_pipe(writer_path, reader_path)
        thread.start_new_thread(self.__receive_thread, (reader_path,))

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__SERVER_OBJECT is None:
            cls.__SERVER_OBJECT = object.__new__(cls)
        return cls.__SERVER_OBJECT

    @staticmethod
    def __get_pipe_default_path(mode: str = 'server') -> tuple:
        if mode == 'server':
            return sys_def.PIPE_READ_PATH, sys_def.PIPE_WRITE_PATH
        return sys_def.PIPE_WRITE_PATH, sys_def.PIPE_READ_PATH

    def __del__(self) -> None:
        self.__exit = True
        time.sleep(self.__EXIT_WAIT_TIME)
        log.info('The stop command of receive thread in %s has been sent!' % self.__mode)
        self.__close_pipe()

    def __close_pipe(self) -> None:
        if self.__pipe_writer is not None:
            os.close(self.__pipe_writer)
            self.__pipe_writer = None
        if self.__pipe_reader is not None:
            os.close(self.__pipe_reader)
            self.__pipe_reader = None

    def __get_path(self, writer_path: str = None, reader_path: str = None) -> tuple:
        if writer_path is None:
            writer_path, _ = self.__get_pipe_default_path(self.__mode)
        if reader_path is None:
            _, reader_path = self.__get_pipe_default_path(self.__mode)
        return writer_path, reader_path

    @staticmethod
    def __create_pipe_file(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
        os.mkfifo(path)

    def __create_file(self, writer_path: str = None, reader_path: str = None) -> None:
        if self.__mode == 'server':
            self.__create_pipe_file(writer_path)
            self.__create_pipe_file(reader_path)

    def __create_sender_pipe(self, writer_path: str = None) -> object:
        pipe_writer = None
        if os.path.exists(writer_path):
            pipe_writer = os.open(writer_path, os.O_SYNC | os.O_CREAT | os.O_RDWR)
            log.info(f'{self.__mode} creates a sender')
        return pipe_writer

    def __create_receiver_pipe(self, reader_path: str = None) -> int:
        pipe_reader = None
        if os.path.exists(reader_path):
            log.info(f'{self.__mode} creates a receiver')
            log.info(f'{self.__mode} is waiting a message')
            pipe_reader = os.open(reader_path, os.O_RDONLY)

        return pipe_reader

    def __create_pipe(self, writer_path: str = None, reader_path: str = None) -> tuple:
        writer_path, reader_path = self.__get_path(writer_path, reader_path)
        self.__create_file(writer_path, reader_path)
        log.info(f"The pipe's path: {self.__mode}, {writer_path} , {reader_path}")

        pipe_writer = self.__create_sender_pipe(writer_path)
        pipe_reader = self.__create_receiver_pipe(reader_path) if self.__mode == 'server' else None

        return pipe_writer, pipe_reader

    def send(self, msg: str) -> None:
        os.write(self.__pipe_writer, msg.encode())

    def receive(self) -> str:
        return self.__msg_queue.get()

    def __receive_thread(self, reader_path) -> None:
        log.info('The receive thread starts!')
        while True:
            if self.__exit:
                log.info(f'The receive thread in {self.__mode} has exited!')
                return

            if self.__pipe_reader is None:
                _, reader_path = self.__get_path(None, reader_path)
                self.__pipe_reader = self.__create_receiver_pipe(reader_path)

            msg = os.read(self.__pipe_reader, self.__BUFF_SIZE)
            if len(msg) == 0:
                time.sleep(self.__THREAD_WAIT_TIME)
                continue

            self.__msg_queue.put(msg.decode())
