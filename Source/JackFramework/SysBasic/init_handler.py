# -*- coding: utf-8 -*-
"""Program bootstrap utilities."""

import os
from typing import Dict

from JackFramework.SysBasic.log_handler import LogHandler as log
from JackFramework.FileHandler.file_handler import FileHandler

from .device_manager import DeviceManager


class InitProgram(object):
    """Prepare directories, logging, and device sanity checks."""

    def __init__(self, args) -> None:
        super().__init__()
        self.__args = args

    def __build_result_directory(self) -> None:
        for path in (self.__args.outputDir, self.__args.modelDir,
                     self.__args.resultImgDir, self.__args.log):
            FileHandler.mkdir(path)

    def __show_args(self) -> None:
        args = self.__args
        log.info('The hyper-parameters are set as follows:')
        ordered_args: Dict[str, object] = {
            'mode': args.mode,
            'dataset': args.dataset,
            'trainListPath': args.trainListPath,
            'valListPath': args.valListPath,
            'outputDir': args.outputDir,
            'modelDir': args.modelDir,
            'resultImgDir': args.resultImgDir,
            'log': args.log,
            'gpu': args.gpu,
            'dist': args.dist,
            'nodes': getattr(args, 'nodes', 1),
            'node_rank': getattr(args, 'node_rank', 0),
            'dataloaderNum': args.dataloaderNum,
            'auto_save_num': args.auto_save_num,
            'sampleNum': args.sampleNum,
            'maxEpochs': args.maxEpochs,
            'batchSize': args.batchSize,
            'lr': args.lr,
            'pretrain': args.pretrain,
            'modelName': args.modelName,
            'imgNum': args.imgNum,
            'valImgNum': args.valImgNum,
            'imgWidth': args.imgWidth,
            'imgHeight': args.imgHeight,
        }

        last_key = next(reversed(ordered_args))
        for key, value in ordered_args.items():
            prefix = '└──' if key == last_key else '├──'
            log.info(f'{prefix} {key}: {value}')

    def __check_paths(self) -> bool:
        validation_map = {
            'trainListPath': (self.__args.trainListPath, os.path.exists, log.error,
                              'the training list does not exist!'),
            'valListPath': (self.__args.valListPath, os.path.exists, log.warning,
                            'the validation list does not exist!'),
        }
        result = True
        log.info('Begin to check the args')
        for _, (target, predicate, reporter, message) in validation_map.items():
            if not predicate(target):
                reporter(message)
                if reporter is log.error:
                    result = False

        for directory in ('outputDir', 'modelDir', 'resultImgDir', 'log'):
            value = getattr(self.__args, directory)
            if os.path.isfile(value):
                log.error(f'A file was passed as `--{directory}`, please pass a directory!')
                result = False

        if result:
            log.info('Finish checking the args')
        else:
            log.info('Error in the process of checking args')
        return result

    def __check_env(self) -> bool:
        log.info('Begin to check the env')
        return DeviceManager.check_cuda(self.__args)

    def init_program(self) -> bool:
        args = self.__args
        self.__build_result_directory()
        log().init_log(args.outputDir, args.pretrain)
        log.info('Welcome to use the JackFramework')
        self.__show_args()
        res = self.__check_paths() and self.__check_env()
        if not res:
            log.error('Failed in the init programs')
        return res
