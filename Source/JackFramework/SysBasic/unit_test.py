# -*- coding: utf-8 -*-
"""Quick smoke tests for progress utilities."""

import time

try:
    from .process_bar import ShowProcess
except ImportError:
    from process_bar import ShowProcess


class SysBasicUnitTestFramework(object):

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _test_process_bar() -> None:
        max_steps = 50
        process_bar = ShowProcess(max_steps, 'Demo', info_done='Finished', bar_width=40)

        for step in range(1, max_steps + 1):
            remaining = max_steps - step
            process_bar.show_process(step, show_info='training', rest_time=remaining * 0.05,
                                     duration=0.05, queue_size=remaining % 8)
            time.sleep(0.05)
        process_bar.close()

    def test(self) -> None:
        self._test_process_bar()


def main() -> None:
    test_framework = SysBasicUnitTestFramework()
    test_framework.test()


if __name__ == '__main__':
    main()
