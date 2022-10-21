# -*- coding: utf-8 -*-
import time

try:
    from .process_bar import ShowProcess
except ImportError:
    from process_bar import ShowProcess


class SysBasicUnitTestFramework(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _test_process_bar() -> None:
        max_steps = 100
        process_bar = ShowProcess(max_steps, 'OK')

        for i in range(max_steps):
            process_bar.show_process(i + 1)
            time.sleep(0.02)
        time.sleep(1)

    def test(self) -> None:
        self._test_process_bar()


def main() -> None:
    test_framework = SysBasicUnitTestFramework()
    test_framework.test()


if __name__ == '__main__':
    main()
