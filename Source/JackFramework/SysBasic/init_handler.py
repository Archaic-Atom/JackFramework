# -*- coding: utf-8 -*-
"""Program bootstrap utilities."""

import os
import warnings
import re
import sys
import atexit
import threading
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

    def __configure_runtime_logging(self) -> None:
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
        args = self.__args

        # Torch C++ logs (covers Gloo/NCCL warnings emitted via PyTorch C++)
        val = env.get('JACK_TORCH_CPP_LOG_LEVEL')
        if val:
            env['TORCH_CPP_LOG_LEVEL'] = val
        elif env.get('JACK_SILENCE_TORCH_CPP') == '1':
            env.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')
        else:
            # Apply sensible defaults based on --debug if user did not set JACK_* or TORCH_CPP_LOG_LEVEL
            if 'TORCH_CPP_LOG_LEVEL' not in env:
                env['TORCH_CPP_LOG_LEVEL'] = 'INFO' if getattr(args, 'debug', False) else 'ERROR'

        # NCCL library logs
        val = env.get('JACK_NCCL_DEBUG')
        if val:
            env['NCCL_DEBUG'] = val
        elif env.get('JACK_SILENCE_NCCL') == '1':
            env.setdefault('NCCL_DEBUG', 'ERROR')
        else:
            if 'NCCL_DEBUG' not in env:
                env['NCCL_DEBUG'] = 'INFO' if getattr(args, 'debug', False) else 'ERROR'

        # Gloo library logs (some c10d prints originate from Gloo directly)
        val = env.get('JACK_GLOO_LOG_LEVEL')
        if val:
            env['GLOO_LOG_LEVEL'] = val
        elif env.get('JACK_SILENCE_GLOO') == '1':
            env.setdefault('GLOO_LOG_LEVEL', 'ERROR')
        else:
            if 'GLOO_LOG_LEVEL' not in env:
                env['GLOO_LOG_LEVEL'] = 'INFO' if getattr(args, 'debug', False) else 'ERROR'

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
        else:
            # Default: suppress the meshgrid indexing deprecation when not debugging
            if not getattr(args, 'debug', False):
                warnings.filterwarnings('ignore', category=UserWarning,
                                        message=r'.*torch\.meshgrid:.*indexing.*')

        # Allow custom warning filters via env if not already provided by user
        if env.get('JACK_PY_WARNINGS') and not env.get('PYTHONWARNINGS'):
            env['PYTHONWARNINGS'] = env['JACK_PY_WARNINGS']

        # Optional: install a stderr filter to drop specific C++ prints that
        # are not controlled by log-level envs (best-effort, terminal only)
        self.__install_stderr_filter()

    def __install_stderr_filter(self) -> None:
        env = os.environ
        debug = getattr(self.__args, 'debug', False)

        # Build suppression patterns
        patterns = []
        # Default: hide Gloo peer connection info when not debugging
        if env.get('JACK_SUPPRESS_GLOO_CONNECT') == '1' or not debug:
            # Match both the short form and the extended form with expected count suffix
            patterns.append(r"^\[Gloo\] Rank \d+ is connected to \d+ peer ranks.*")
        # Optional: hide NCCL destroy_process_group timing warning
        if env.get('JACK_SUPPRESS_NCCL_DESTROY_WARNING') == '1':
            patterns.append(r"ProcessGroupNCCL\.cpp:.*destroy_process_group\(\).*")
        # User-provided additional regex filters (separated by '|')
        user_filter = env.get('JACK_STDERR_FILTER')
        if user_filter:
            for pat in user_filter.split('|'):
                pat = pat.strip()
                if pat:
                    patterns.append(pat)

        if not patterns:
            return

        try:
            compiled = [re.compile(p) for p in patterns]
        except re.error:
            # Invalid regex; do nothing
            return

        # Install filter only for stderr to avoid interfering with live stdout
        # rendering (e.g., progress bars) and TTY colour detection.
        self.__install_stream_filter(sys.stderr, compiled, name='stderr')

    @staticmethod
    def __install_stream_filter(stream, compiled_patterns, name: str = 'stderr') -> None:
        if not hasattr(stream, 'fileno'):
            return
        try:
            orig_fd = os.dup(stream.fileno())
        except Exception:
            return

        r_fd, w_fd = os.pipe()
        try:
            os.dup2(w_fd, stream.fileno())
        except Exception:
            os.close(r_fd)
            os.close(w_fd)
            os.close(orig_fd)
            return

        stop_flag = threading.Event()

        def _pump():
            buf = b''
            try:
                while not stop_flag.is_set():
                    try:
                        chunk = os.read(r_fd, 4096)
                        if not chunk:
                            break
                        buf += chunk
                        while b'\n' in buf:
                            line, buf = buf.split(b'\n', 1)
                            text = line.decode(errors='ignore')
                            if any(p.search(text) for p in compiled_patterns):
                                continue
                            os.write(orig_fd, line + b'\n')
                    except InterruptedError:
                        continue
            finally:
                if buf:
                    text = buf.decode(errors='ignore')
                    if not any(p.search(text) for p in compiled_patterns):
                        os.write(orig_fd, buf)
                os.close(r_fd)
                os.close(orig_fd)

        t = threading.Thread(target=_pump, name=f'jf-{name}-filter', daemon=True)
        t.start()

        def _cleanup():
            try:
                stop_flag.set()
                try:
                    os.close(w_fd)
                except Exception:
                    pass
            except Exception:
                pass

        atexit.register(_cleanup)

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

    # Expose a helper so spawned workers can apply logging knobs early
    @staticmethod
    def apply_runtime_logging(args: object) -> None:
        try:
            helper = InitProgram(args)
            helper._InitProgram__configure_runtime_logging()
        except Exception:
            # Best-effort; do not crash on logging configuration
            pass
