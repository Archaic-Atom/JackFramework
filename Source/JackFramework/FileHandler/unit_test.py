# -*- coding: utf-8 -*-
import os
import sys

try:
    from .file_handler import FileHandler
    from .model_saver import ModelSaver
except ImportError:
    from file_handler import FileHandler
    from model_saver import ModelSaver


class FileHandlerUnitTestFramework(object):
    CHECK_POINT_LIST_NAME = 'checkpoint.list'
    LAST_MODEL_NAME = 'last model name:'

    def __init__(self):
        super().__init__()

    @staticmethod
    def _test_model_saver() -> None:
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../JackFramework'))
        print(sys.path)
        ModelSaver.write_check_point_list('./Checkpoint/', 'test_model_1_epoch_100.pth')

    def _test_file_handler(self) -> None:
        file_dir = './Checkpoint/'
        file_name = 'checkpoint.list'
        fd_checkpoint_list = FileHandler.open_file(file_dir + file_name)
        str_line = FileHandler.get_line_fd(fd_checkpoint_list, 0)
        print(str_line)
        test_file_name = "test_model_1_epoch_%d.pth"

        # Checkpoint's list file
        for i in range(50):
            fd_checkpoint_list = FileHandler.open_file(file_dir + self.CHECK_POINT_LIST_NAME)
            str_line = FileHandler.get_line_fd(fd_checkpoint_list, 0)
            file_name = test_file_name % i
            if str_line[: len(self.LAST_MODEL_NAME)] != self.LAST_MODEL_NAME:
                FileHandler.close_file(fd_checkpoint_list)
                fd_checkpoint_list = None
                os.remove(file_dir + self.CHECK_POINT_LIST_NAME)

            if fd_checkpoint_list is None:
                fd_checkpoint_list = FileHandler.open_file(file_dir + self.CHECK_POINT_LIST_NAME)
                FileHandler.write_file(fd_checkpoint_list, self.LAST_MODEL_NAME + file_name)
                FileHandler.write_file(fd_checkpoint_list, file_name)
                FileHandler.close_file(fd_checkpoint_list)
            else:
                fd_checkpoint_temp_list = FileHandler.open_file(
                    file_dir + self.CHECK_POINT_LIST_NAME + '.temp')
                FileHandler.write_file(fd_checkpoint_temp_list, self.LAST_MODEL_NAME + file_name)
                FileHandler.copy_file(fd_checkpoint_list, fd_checkpoint_temp_list, 1)
                FileHandler.write_file(fd_checkpoint_temp_list, file_name)
                FileHandler.close_file(fd_checkpoint_list)
                FileHandler.close_file(fd_checkpoint_temp_list)
                os.remove(file_dir + self.CHECK_POINT_LIST_NAME)
                os.rename(file_dir + self.CHECK_POINT_LIST_NAME + '.temp',
                          file_dir + self.CHECK_POINT_LIST_NAME)

    def test(self) -> None:
        self._test_model_saver()
        self._test_file_handler()


def main() -> None:
    test_framework = FileHandlerUnitTestFramework()
    test_framework.test()


if __name__ == '__main__':
    main()
