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
| `--batchSize` | 64 | Per-device batch size. |
| `--maxEpochs` | 100 | Training epochs. |
| `--auto_save_num` | 1 | Checkpoint frequency (epochs). Set `0` to disable. |
| `--trainListPath` / `--valListPath` | CSV stubs | Dataset manifest locations. |
| `--outputDir`, `--modelDir`, `--resultImgDir`, `--log` | ./Result/ etc. | Output folders are created automatically. |
| `--debug` | `False` | Extra logging hints (e.g., unused parameters during DDP). |

Run `python your_entry.py --help` to see the full list (plus any custom flags you add in `user_parser`).

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

## Changelog
- **2024-09-15** (current): runtime validation hardened, device/mode guards improved, README refreshed for the PyTorch 2.4 stack, version synced to `0.1.1`.
- **2021-07-01**: initial public README, GitHub Actions, installer scripts.
- **2021-05-28**: project bootstrap, packaging script.

JackFramework is released under the MIT License.
