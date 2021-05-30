# -*- coding: UTF-8 -*-
import sys
import time


class ShowProcess():
    _INIT_COUNTER = 0
    _MAX_STEP = 0
    _MAX_ARROW = 5
    _EACH_STEP = 1
    _MAX_SCORE = 100.0
    _INFO_DONE = 'done'

    def __init__(self, max_steps: int, info: str = '', info_done='Done') -> object:
        self.__info = info
        self.__max_steps = max_steps
        self.__counter = self._INIT_COUNTER
        self.__info_done = info_done

    # [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, start_counter: int = None,
                     show_info: str = '',
                     rest_time: str = '',
                     duration: str = '',
                     queue_size: str = ''):
        self.__count(start_counter)
        num_arrow, num_line, percent = self.__cal_bar()

        info_done = self.__genearte_info_done()
        queue_size = self.__generate_queue_size(queue_size)
        rest_time = self.__generate_rest_time(rest_time, duration)

        process_str = self.__generate_show_data(num_arrow, num_line,
                                                percent, show_info,
                                                info_done, queue_size,
                                                rest_time)
        self.__print(process_str)

    def close(self):
        print('')
        self.__counter = self._INIT_COUNTER

    def __count(self, start_counter: int) -> None:
        if start_counter is not None:
            self.__counter = start_counter
        else:
            self.__counter += ShowProcess._EACH_STEP

    def __cal_bar(self) -> int:
        num_arrow = int(self.__counter * self._MAX_ARROW / self.__max_steps)
        num_line = self._MAX_ARROW - num_arrow
        percent = self.__counter * self._MAX_SCORE / self.__max_steps
        return num_arrow, num_line, percent

    def __genearte_info_done(self) -> str:
        if self.__counter >= self.__max_steps:
            info_done = ', ' + self.__info_done
        else:
            info_done = ''
        return info_done

    @staticmethod
    def __generate_queue_size(queue_size: int) -> str:
        if queue_size != '':
            queue_size = '(qs: %d), ' % queue_size

        return queue_size

    @staticmethod
    def __generate_rest_time(rest_time: int, duration: int) -> str:
        if rest_time != '':
            rest_time = '(rt: %.3f s' % rest_time

        if duration != '':
            rest_time = rest_time + ', bs: %.3f s)' % duration
        else:
            rest_time = rest_time + ')'

        return rest_time

    def __generate_show_data(self, num_arrow: int, num_line: int,
                             percent: float, show_info: str,
                             info_done: str, queue_size: str,
                             rest_time: str) -> str:
        process_str = '[' + self.__info + '] [' + '>' * num_arrow   \
            + '-' * num_line + ']'                                  \
            + ' %d / %d, ' % (self.__counter, self.__max_steps)     \
            + '%.2f' % percent + '%' + ' '                          \
            + show_info + ' ' + queue_size                          \
            + rest_time + info_done                                 \
            + '\r'
        return process_str

    @staticmethod
    def __print(process_str: str) -> None:
        sys.stdout.write(process_str)
        sys.stdout.flush()

    def __check_finish(self):
        if self.__counter >= self.__max_steps:
            self.close()


if __name__ == '__main__':
    max_steps = 50

    process_bar = ShowProcess(max_steps, 'OK')

    for i in range(max_steps):
        process_bar.show_process(i + 1)
        time.sleep(0.01)
    time.sleep(50)
