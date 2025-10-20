# -*- coding: utf-8 -*-
"""Program bootstrap utilities."""

import os
import warnings
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

    @staticmethod
    def __configure_runtime_logging() -> None:
        """Apply environment-driven logging and warning controls.

        Supported env vars (set before init):
        - JACK_TORCH_CPP_LOG_LEVEL: one of TRACE|DEBUG|INFO|WARN|ERROR (sets TORCH_CPP_LOG_LEVEL)
        - JACK_SILENCE_TORCH_CPP=1: force TORCH_CPP_LOG_LEVEL=ERROR
        - JACK_NCCL_DEBUG: one of TRACE|INFO|WARN|ERROR (sets NCCL_DEBUG)
        - JACK_SILENCE_NCCL=1: force NCCL_DEBUG=ERROR
        - JACK_NCCL_DEBUG_FILE: redirect NCCL logs to the given file (sets NCCL_DEBUG_FILE)
        - JACK_SILENCE_PY_WARNINGS=1: ignore all Python warnings
        - JACK_SILENCE_TORCH_WARNINGS=1: ignore torch.* UserWarning
        - JACK_SUPPRESS_MESHGRID_WARNING=1: suppress meshgrid indexing deprecation warning
        - JACK_PY_WARNINGS: pass-through to PYTHONWARNINGS if not already set
        """

        env = os.environ

        # Torch C++ logs (covers Gloo/NCCL warnings emitted via PyTorch C++)
        val = env.get('JACK_TORCH_CPP_LOG_LEVEL')
        if val:
            env['TORCH_CPP_LOG_LEVEL'] = val
        elif env.get('JACK_SILENCE_TORCH_CPP') == '1':
            env.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')

        # NCCL library logs
        val = env.get('JACK_NCCL_DEBUG')
        if val:
            env['NCCL_DEBUG'] = val
        elif env.get('JACK_SILENCE_NCCL') == '1':
            env.setdefault('NCCL_DEBUG', 'ERROR')

        if env.get('JACK_NCCL_DEBUG_FILE'):
            env['NCCL_DEBUG_FILE'] = env['JACK_NCCL_DEBUG_FILE']

        # Python warnings
        if env.get('JACK_SILENCE_PY_WARNINGS') == '1':
            warnings.filterwarnings('ignore')

        if env.get('JACK_SILENCE_TORCH_WARNINGS') == '1':
            warnings.filterwarnings('ignore', category=UserWarning, module=r'^torch')

        if env.get('JACK_SUPPRESS_MESHGRID_WARNING') == '1':
            warnings.filterwarnings('ignore', category=UserWarning,
                                    message=r'.*torch\.meshgrid:.*indexing.*')

        # Allow custom warning filters via env if not already provided by user
        if env.get('JACK_PY_WARNINGS') and not env.get('PYTHONWARNINGS'):
            env['PYTHONWARNINGS'] = env['JACK_PY_WARNINGS']

    def init_program(self) -> bool:
        args = self.__args
        # Apply runtime logging knobs (must be early)
        self.__configure_runtime_logging()
        self.__build_result_directory()
        log().init_log(args.outputDir, args.pretrain)
        log.info('Welcome to use the JackFramework')
        self.__show_args()
        res = self.__check_paths() and self.__check_env()
        if not res:
            log.error('Failed in the init programs')
        return res
