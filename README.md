![Build](https://github.com/Archaic-Atom/JackFramework/actions/workflows/build%20test.yml/badge.svg?event=push)
![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg?style=plastic)
![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-orange.svg?style=plastic)
![cuDNN 9.1](https://img.shields.io/badge/cuDNN-9.1-blue.svg?style=plastic)
![MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

# JackFramework
A lightweight, production-friendly orchestration layer on top of PyTorch. JackFramework standardizes model/data wiring, distributed execution, logging, and persistence so you can focus on modelling and experimentation — not boilerplate.

- Template project: https://github.com/Archaic-Atom/Template-jf
- Contact: raoxi36@foxmail.com

## Table of Contents
- Overview
- Features
- Requirements
- Quick Start
- Installation
- Using JackFramework
- Execution Modes
- Distributed Training
- Observability & Logging
- Project Structure
- Templates & Examples
- Troubleshooting
- Changelog
- License

## Overview
JackFramework wraps common training infrastructure concerns (device management, DDP/DP, logging, progress bars, checkpointing, and mode control) behind a small set of clear interfaces, leaving you to implement just your model and data logic.

## Features
- PyTorch 2.x ready with both DataParallel and DistributedDataParallel paths.
- Clean separation of user code via `ModelHandlerTemplate`, `DataHandlerTemplate`, and `NetWorkInferenceTemplate`.
- Multiple execution modes (`train`, `test`, `background`, `web`) via a unified application entrypoint.
- Rich observability: colourised logs, TensorBoard scalars, progress bars, resumable checkpoints.
- Safer runtime: explicit argument validation and defensive error handling.

## Requirements
| Component | Recommended |
|-----------|-------------|
| OS        | Linux 16.04+ with CUDA-capable GPUs |
| Python    | 3.10 (matches `environment.yml`) |
| PyTorch   | 2.4.1 with CUDA 11.8 / cuDNN 9.1 |
| Optional  | TensorBoard for visualisation, Django for the web mode |

For a reproducible environment use the included Conda spec (`environment.yml`).

## Quick Start
```python
from JackFramework import Application
from your_project.interface import UserInterface

if __name__ == '__main__':
    Application(UserInterface(), application_name='StereoDepth').start()
```
Run modes by switching `--mode <train|test|background|web>`.

## Installation
```bash
# create and activate the suggested environment
conda env create -f environment.yml
conda activate JackFramework-torch2.3.1

# install JackFramework into the active environment
./install.sh

# sanity check
python -c "import JackFramework as jf; print(jf.version())"
```
Clean artifacts (logs, checkpoints, build outputs) with `./clean.sh`.

## Using JackFramework
1. Implement your interface by subclassing `NetWorkInferenceTemplate`.
2. Return your `ModelHandlerTemplate` and `DataHandlerTemplate` from `inference`.
3. Optionally extend the CLI in `user_parser` for domain-specific flags.

These templates provide model construction, optimizers/schedulers, loss/metric computation, data loading/splitting, and result persistence hooks.

## Execution Modes
| Mode | Description |
| ---- | ----------- |
| `train` | Training with optional validation. Supports DP/DDP, TensorBoard, auto checkpoints. |
| `test` | Restore latest checkpoint and evaluate/dump results. |
| `background` | Pipe-driven inference server (single GPU, batch size 1). |
| `web` | Launch the bundled Django server (see `args.web_cmd`) for browser demos. |

Switch modes via `--mode <train|test|background|web>` when invoking your entry script.

## Distributed Training
- Single GPU
  ```bash
  python your_entry.py --dist false --gpu 1
  ```
- Single node, multi-GPU (framework default via `mp.spawn`)
  ```bash
  python your_entry.py --dist true --gpu 4
  ```
- Single node, multi-GPU (torchrun)
  ```bash
  torchrun --nproc_per_node=4 your_entry.py --dist true --gpu 4
  ```
- Multi node, multi-GPU (torchrun on each node)
  ```bash
  torchrun --nnodes=2 --nproc_per_node=4 \
    --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=29500 \
    your_entry.py --dist true --gpu 4 --nodes 2 --node_rank ${NODE_RANK}
  ```
  The framework reuses `RANK` / `LOCAL_RANK` / `WORLD_SIZE`; only rank 0 prints to the terminal by default, while all ranks write `output.log`.

Common CLI flags
| Flag | Default | Notes |
|------|---------|-------|
| `--gpu` | 2 | Number of GPUs. Use `0` for CPU. |
| `--dist` | `True` | Enable DDP. Falls back to DP/CPU when GPUs < requested. |
| `--nodes` | 1 | Number of nodes (for multi-node DDP). |
| `--node_rank` | 0 | Rank of this node (0-based). |
| `--batchSize` | 64 | Per-device batch size. |
| `--maxEpochs` | 100 | Training epochs. |
| `--auto_save_num` | 1 | Checkpoint frequency (epochs). Set `0` to disable. |
| `--trainListPath` / `--valListPath` | CSV | Dataset manifest paths. |
| `--outputDir`, `--modelDir`, `--resultImgDir`, `--log` | ./Result/ | Output folders. |
| `--debug` | `False` | Extra logging hints (e.g., unused params in DDP). |

## Observability & Logging
- TensorBoard writes to `--log`; launch with `tensorboard --logdir <log_dir>`.
- Progress bars auto-adjust to terminal width; override via `JF_PROGRESS_COLUMNS`.
- Environment variables
  - `JACK_LOG_ALL_RANKS=1`: print console logs from every rank (default: only rank 0). File logging is always enabled for all ranks.
    - torchrun example: `JACK_LOG_ALL_RANKS=1 torchrun --nproc_per_node=4 your_entry.py --dist true --gpu 4`
    - spawn example: `JACK_LOG_ALL_RANKS=1 python your_entry.py --dist true --gpu 4`
  - `MASTER_ADDR` / `MASTER_PORT`: rendezvous endpoint for DDP.
  - `RANK`, `LOCAL_RANK`, `WORLD_SIZE`: honoured for torchrun/elastic launches.
  - Torch C++ log filtering (covers c10d/Gloo/NCCL messages printed by PyTorch):
    - `JACK_TORCH_CPP_LOG_LEVEL=ERROR` (or `DEBUG|INFO|WARN|TRACE`) — sets `TORCH_CPP_LOG_LEVEL`
    - `JACK_SILENCE_TORCH_CPP=1` — shortcut for `TORCH_CPP_LOG_LEVEL=ERROR`
  - Gloo library logs (some c10d messages originate from Gloo directly):
    - `JACK_GLOO_LOG_LEVEL=ERROR` (or `WARN|INFO|DEBUG|TRACE`) — sets `GLOO_LOG_LEVEL`
    - `JACK_SILENCE_GLOO=1` — shortcut for `GLOO_LOG_LEVEL=ERROR`
  - NCCL library logs:
    - `JACK_NCCL_DEBUG=ERROR` (or `WARN|INFO|TRACE`) — sets `NCCL_DEBUG`
    - `JACK_SILENCE_NCCL=1` — shortcut for `NCCL_DEBUG=ERROR`
    - `JACK_NCCL_DEBUG_FILE=/path/to/nccl.log` — redirect NCCL logs to file
  - Python warnings:
    - `JACK_SILENCE_PY_WARNINGS=1` — ignore all Python warnings
    - `JACK_SILENCE_TORCH_WARNINGS=1` — ignore `torch.*` UserWarning (e.g., meshgrid deprecations)
    - `JACK_SUPPRESS_MESHGRID_WARNING=1` — ignore the specific `torch.meshgrid(..., indexing=...)` deprecation warning
    - Advanced: use `PYTHONWARNINGS` directly (e.g., `PYTHONWARNINGS=ignore:::torch.functional`)

Debug flag
- `--debug true` sets sensible verbose defaults if you do not specify env vars:
  - `TORCH_CPP_LOG_LEVEL=INFO`, `NCCL_DEBUG=INFO`, and Python warnings shown.
- `--debug false` (default) applies quieter defaults:
  - `TORCH_CPP_LOG_LEVEL=ERROR`, `NCCL_DEBUG=ERROR`, and suppress `torch.meshgrid` indexing warning.
- Any `JACK_*` or low-level env vars you set explicitly will take precedence over these defaults.

## Project Structure
```
.
├── Source
│   ├── JackFramework
│   │   ├── Core            # Application modes, graph builders, schedulers
│   │   ├── Evaluation      # Loss/metric helpers
│   │   ├── FileHandler     # I/O utilities (checkpoints, TensorBoard, pipes)
│   │   ├── ImgHandler      # Augmentation and image I/O helpers
│   │   ├── NN              # Building blocks, layers, ops
│   │   ├── SysBasic        # Args parsing, logging, devices, progress bars
│   │   ├── Tools           # Misc tooling (named pipes, shell helpers)
│   │   └── UserTemplate    # Templates for custom models + dataloaders
│   ├── setup.py
│   └── ...
├── environment.yml
├── install.sh / clean.sh / build.sh
├── LICENSE
└── README.md
```

## Templates & Examples
- Framework Template: https://github.com/Archaic-Atom/Template-jf
- Demo Project: https://github.com/Archaic-Atom/Demo-jf

## Troubleshooting
- Fast failures usually mean argument or template issues; revisit your implementations and CLI.
- DDP launches require consistent `--ip`, `--port`, `--gpu`. Port collisions are auto-probed and logged.
- Named-pipe modes (`background`) require a POSIX environment.
- About NCCL “destroy_process_group was not called” warnings: JackFramework explicitly tears down DDP across ranks at process exit. The warning may still appear on stderr in rare timing cases and can be ignored if every rank logs destruction as expected. You can temporarily silence C++ warnings via `TORCH_CPP_LOG_LEVEL=ERROR`.

## Changelog
- 2025-09-18
  - Hardened runtime validation across graph/mode/device helpers.
  - Synced packaging version info (`0.1.1`).
  - Refined install/build/clean scripts.
  - Auto-load metadata in `setup.py`.
  - Progress bar rendering improvements.
- 2025-09-15
  - README refreshed for the PyTorch 2.4 toolchain.
  - Defensive checks around NN blocks.
- 2021-07-01
  - Added GitHub Actions CI and first README.
  - Installation helper scripts.
- 2021-05-28
  - Project bootstrap.

## License
MIT License
