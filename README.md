[![build test](https://github.com/Archaic-Atom/JackFramework/actions/workflows/build%20test.yml/badge.svg?event=push)](https://github.com/Archaic-Atom/JackFramework/actions/workflows/build%20test.yml)
![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg?style=plastic)
![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-orange.svg?style=plastic)
![cuDNN 9.1](https://img.shields.io/badge/cuDNN-9.1-blue.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

> JackFramework is a lightweight training orchestration layer on top of PyTorch. It standardises data/model wiring, distributed execution, logging and persistence so you can focus on modelling. A runnable template project lives at https://github.com/Archaic-Atom/Template-jf. Questions? raoxi36@foxmail.com

## Highlights
- PyTorch 2.x ready with both DataParallel and DistributedDataParallel paths.
- Clean separation between user code and framework glue through `ModelHandlerTemplate`, `DataHandlerTemplate`, and `NetWorkInferenceTemplate`.
- Multiple execution modes (`train`, `test`, `background`, `web`) controlled by a unified application entrypoint.
- Rich observability: colourised logging, TensorBoard scalars, progress bars, resumable checkpoints.
- Safer runtime: explicit argument validation, defensive error handling, and clearer failure modes after the latest refactor.

## Requirements
| Component | Recommended |
|-----------|-------------|
| OS        | Linux 16.04+ with CUDA-capable GPUs |
| Python    | 3.10 (matches `environment.yml`) |
| PyTorch   | 2.4.1 with CUDA 11.8 / cuDNN 9.1 |
| Optional  | TensorBoard for visualisation, Django for the web mode |

For an exact reproducible environment use the provided Conda spec (`environment.yml`).

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

To remove generated artifacts (logs, checkpoints, build outputs) run `./clean.sh`.

## Usage Overview
1. **Implement the user interface** by subclassing `NetWorkInferenceTemplate`. Return your `ModelHandlerTemplate` and `DataHandlerTemplate` implementations from `inference`, and extend the CLI parser if you need custom flags.
2. **Instantiate the framework** with your interface and kick off a mode:
    ```python
    from JackFramework import Application
    from your_project.interface import UserInterface

    if __name__ == '__main__':
        Application(UserInterface(), application_name='StereoDepth').start()
    ```
3. **Provide dataset/model handlers** by extending the templates in `JackFramework/UserTemplate`. These supply model construction, optimisation, loss/metric computation, data loading, and result persistence hooks.

### Supported Modes
| Mode | Description |
| ---- | ----------- |
| `train` | Runs the training + optional validation loop, supports DP/DDP, TensorBoard, auto checkpointing. |
| `test` | Restores the latest checkpoint and performs evaluation / result dumping. |
| `background` | Pipe-driven inference server (single GPU, batch size 1) backed by named pipes. |
| `web` | Boots the bundled Django server (see `args.web_cmd`) for browser-based demos. |

Switch modes via `--mode <train|test|background|web>` when invoking your entry script.

### Frequently Used CLI Flags
| Flag | Default | Notes |
|------|---------|-------|
| `--gpu` | 2 | Number of GPUs. Set to `0` for CPU. |
| `--dist` | `True` | Enable DistributedDataParallel. Falls back to DP/CPU when GPUs < requested. |
| `--nodes` | 1 | Number of nodes participating in distributed execution. |
| `--node_rank` | 0 | Rank of this node in the multi-node job (0-based). |
| `--batchSize` | 64 | Per-device batch size. |
| `--maxEpochs` | 100 | Training epochs. |
| `--auto_save_num` | 1 | Checkpoint frequency (epochs). Set `0` to disable. |
| `--trainListPath` / `--valListPath` | CSV stubs | Dataset manifest locations. |
| `--outputDir`, `--modelDir`, `--resultImgDir`, `--log` | ./Result/ etc. | Output folders are created automatically. |
| `--debug` | `False` | Extra logging hints (e.g., unused parameters during DDP). |

Run `python your_entry.py --help` to see the full list (plus any custom flags you add in `user_parser`).

**Environment Tweaks**
- `JF_PROGRESS_COLUMNS`: override the detected terminal width (useful for `nohup`/non-TTY runs) so the progress bar can expand to the specified column count.
- `MASTER_ADDR` / `MASTER_PORT`: override the rendezvous endpoint used by distributed jobs (defaults to `--ip` / `--port`).
- `RANK`, `LOCAL_RANK`, `WORLD_SIZE`: honoured when launching with `torchrun`/elastic training; set automatically for single-node launches.
- `JACK_LOG_ALL_RANKS`: set to `1` to enable console logs from every rank (by default only rank 0 prints to the terminal). File logging is always enabled for all ranks.
  - Example (torchrun): `JACK_LOG_ALL_RANKS=1 torchrun --nproc_per_node=4 your_entry.py --dist true --gpu 4`
  - Example (single node spawn): `JACK_LOG_ALL_RANKS=1 python your_entry.py --dist true --gpu 4`

### Distributed Launch Notes
- **Single GPU**
  ```bash
  python your_entry.py --dist false --gpu 1
  ```
- **Single node, multi-GPU** (framework default via `mp.spawn`):
  ```bash
  python your_entry.py --dist true --gpu 4
  ```
- **Single node, multi-GPU (torchrun)**: to align with multi-node launch workflow:
  ```bash
  torchrun --nproc_per_node=4 your_entry.py --dist true --gpu 4
  ```
- **Multi node, multi-GPU**: use `torchrun` (or any launcher that populates `RANK` / `LOCAL_RANK` / `WORLD_SIZE`) on every node:
  ```bash
  torchrun --nnodes=2 --nproc_per_node=4 \
    --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=29500 \
    your_entry.py --dist true --gpu 4 --nodes 2 --node_rank ${NODE_RANK}
  ```
  The framework reuses these environment variables; only rank 0 prints console logs while every rank still writes to `output.log`.

## Project Layout
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
- **Framework Template**: https://github.com/Archaic-Atom/Template-jf (recommended starting point for new projects).
- **Demo Project**: https://github.com/Archaic-Atom/Demo-jf (shows an end-to-end training pipeline).

## Troubleshooting & Tips
- JackFramework now performs explicit input validation (assertions were removed). If something fails fast, revisit your arguments and template implementations.
- Distributed launches rely on `--ip`, `--port`, and `--gpu`. When port collisions are detected we auto-probe free ports and log the choice.
- TensorBoard logs write to `--log`. Launch with `tensorboard --logdir <log_dir>`.
- Named-pipe based modes (`background`) require a POSIX environment.

- **2025-09-18**
  - Hardened runtime validation (replaced assertions with explicit errors across graph/mode/device helpers).
  - Synchronized versioning in code and packaging (`0.1.1`).
  - Refined installation/build/clean scripts for pip-based workflows and safer defaults.
  - Refactored `setup.py` to auto-load version and README metadata.
  - Refresh progress bar rendering (terminal-aware truncation + dynamic width detection).
- **2025-09-15**
  - README refreshed for the PyTorch 2.4 toolchain.
  - Initial defensive checks after refactoring the NN blocks.
- **2021-07-01**
  - Added GitHub Actions CI and first public README iteration.
  - Installation helper scripts introduced.
- **2021-05-28**
  - Project bootstrap with packaging scaffold.

JackFramework is released under the MIT License.
