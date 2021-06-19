# -*- coding: utf-8 -*-
import os
import torch

import JackFramework.SysBasic.define as sysdefine
from JackFramework.SysBasic.loghander import LogHandler as log
from JackFramework.FileHandler.filehandler import FileHandler
from .filehandler import FileHandler


class ModelSaver(object):
    """docstring for ModelSaver"""
    ROW_ONE = 1
    FIST_SAVE_TIME = True

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_check_point_path(path: str) -> str:
        checkpoint_path = None
        if os.path.isfile(path):
            log.info("Begin loading checkpoint from this file: '{}'".format(path))
            checkpoint_path = path
        else:
            log.info("Begin loading checkpoint from this folder: '{}'".format(path))
            checkpoint_list_file = path + sysdefine.CHECK_POINT_LIST_NAME
            if not os.path.isfile(checkpoint_list_file):
                log.warning("We don't find the checkpoint list file: '{}'!".format(
                    checkpoint_list_file))
            else:
                checkpoint_path = path + ModelSaver.__get_model_name(checkpoint_list_file)
                log.info("Get the path of model: '{}'".format(checkpoint_path))

        return checkpoint_path

    @staticmethod
    def load_model(model: object, file_path: str, rank: object = None) -> None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if rank is not None else None
        checkpoint = torch.load(file_path, map_location)

        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
        # ignore_block = ['module.feature_extraction.block_1']

        # for k, v in checkpoint['state_dict'].items():
        #    print(k)

        # pretrained_dict = {k: v for k,
        #                   v in checkpoint['state_dict'].items() if k not in ignore_block}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict, strict=True)

        model.load_state_dict(checkpoint['state_dict'], strict=True)
        log.info("Model loaded successfully")

    @staticmethod
    def load_opt(opt: object, file_path: str, rank: object = None) -> None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if rank is not None else None
        checkpoint = torch.load(file_path, map_location)

        # opt_dict = opt.state_dict()
        # pretrained_dict = {k: v for k, v in checkpoint['optimizer'].items() if k in opt_dict}
        # print(pretrained_dict)
        # opt_dict.update(pretrained_dict)
        # opt.load_state_dict(opt_dict)

        # opt.load_state_dict(checkpoint['optimizer'])
        log.info("opt loaded successfully")

    @staticmethod
    def save(file_dir: str, file_name: str, model_dict: dict) -> None:
        torch.save(model_dict, file_dir + file_name)
        log.info("Save model in : '{}'".format(file_dir + file_name))
        ModelSaver.write_check_point_list(file_dir, file_name)

    @staticmethod
    def __get_model_name(file_path: str) -> str:
        str_line = FileHandler.get_line(file_path, ModelSaver.ROW_ONE)
        return str_line[len(sysdefine.LAST_MODEL_NAME):len(str_line)]

    @staticmethod
    def write_check_point_list(file_dir: str, file_name: str) -> None:
        fd_checkpoint_list = None
        if ModelSaver.FIST_SAVE_TIME:
            if os.path.isfile(file_dir + sysdefine.CHECK_POINT_LIST_NAME):
                os.remove(file_dir + sysdefine.CHECK_POINT_LIST_NAME)
            ModelSaver.FIST_SAVE_TIME = False
        else:
            fd_checkpoint_list = FileHandler.open_file(file_dir + sysdefine.CHECK_POINT_LIST_NAME)
            str_line = FileHandler.get_line_fd(fd_checkpoint_list, 0)
            if str_line[0:len(sysdefine.LAST_MODEL_NAME)] != sysdefine.LAST_MODEL_NAME:
                log.warning("The checklist file is wrong! We will rewrite this file")
                FileHandler.close_file(fd_checkpoint_list)
                fd_checkpoint_list = None
                os.remove(file_dir + sysdefine.CHECK_POINT_LIST_NAME)

        if fd_checkpoint_list is None:
            fd_checkpoint_list = FileHandler.open_file(file_dir + sysdefine.CHECK_POINT_LIST_NAME)
            FileHandler.write_file(fd_checkpoint_list, sysdefine.LAST_MODEL_NAME + file_name)
            FileHandler.write_file(fd_checkpoint_list, file_name)
            FileHandler.close_file(fd_checkpoint_list)
        else:
            fd_checkpoint_temp_list = FileHandler.open_file(file_dir + sysdefine.CHECK_POINT_LIST_NAME + '.temp')
            FileHandler.write_file(fd_checkpoint_temp_list,
                                   sysdefine.LAST_MODEL_NAME + file_name)
            FileHandler.copy_file(fd_checkpoint_list, fd_checkpoint_temp_list, 1)
            FileHandler.write_file(fd_checkpoint_temp_list, file_name)
            FileHandler.close_file(fd_checkpoint_list)
            FileHandler.close_file(fd_checkpoint_temp_list)
            os.remove(file_dir + sysdefine.CHECK_POINT_LIST_NAME)
            os.rename(file_dir + sysdefine.CHECK_POINT_LIST_NAME + '.temp', file_dir + sysdefine.CHECK_POINT_LIST_NAME)


def debug_main():
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../JackFramework'))
    print(sys.path)
    TEST_FILE_DIR = './Checkpoint/'
    MODEL_FILE_NAME = 'test_model_1_epoch_100.pth'
    ModelSaver.write_check_point_list(TEST_FILE_DIR, MODEL_FILE_NAME)


if __name__ == '__main__':
    debug_main()
