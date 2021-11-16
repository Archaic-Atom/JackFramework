[![build test](https://github.com/Archaic-Atom/JackFramework/actions/workflows/build%20test.yml/badge.svg?event=push)](https://github.com/Archaic-Atom/JackFramework/actions/workflows/build%20test.yml)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.7](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDnn 7.3.6](https://img.shields.io/badge/cudnn-7.3.6-green.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

>This is a training framework based on pythorch, which is used to rapidly build the model, without caring about the training process (such as DDP or DP, Tensorboard, et al.). The demo for use can be find in FrameworkTemplate (https://github.com/Archaic-Atom/FrameworkTemplate). if you have any questions, please send an e-mail to raoxi36@foxmail.com

# add one people
--- 
#### 0. Software Environment
**1) OS Environment**
```
$ os >= linux 16.04
$ cudaToolKit >= 10.1
$ cudnn >= 7.3.6
```

**2) Python Environment**
```
$ python == 3.8.5
$ pythorch >= 1.15.0
$ numpy == 1.14.5
$ opencv == 3.4.0
$ PIL == 5.1.0
```

---
#### 1. Hardware Environment
This framework is only used in GPUs.

---
#### 2. How to use our framework:
**1) Build env**
```
$ conda env create -f environment.yml
$ conda activate JackFramework-torch1.7.1
```
**2) Install the JackFramework lib**
```
$ ./install.sh
```
**3) Check the version (optional)**
```
$ python -c "import JackFramework as jf; print(jf.version())"
```

**4) the template for using the JackFramework**

you can find the template project in: https://github.com/Archaic-Atom/FameworkTemplate

**Related Arguments for training or testing process**
|   Args        |   Type  |      Description                 | Default         |
|:-------------:|:-------:|:--------------------------------:|:---------------:|
| mode          |  [str]  |         train or test            |  train          |
| gpu           |  [int]  |        the number of gpus        |    2            |
| auto_save_num |  [int]  | the number of interval save      |    1            |
| dataloaderNum |  [int]  |  the number of dataloders        |    8            |
| pretrain      |  [bool] |    is a new traning process>     |  False          |
| ip            |  [str]  | used for distributed training    | 127.0.0.1       |
| port          |  [str]  | used for distributed training    | 8086            |
| dist          |  [bool] | distributed training (DDP)       | True            |
| trainListPath |  [str]  | the list for training or testing | ./Datasets/*.csv|
| valListPath   |  [str]  | the list for validate process    | ./Datasets/*.csv|
| outputDir     |  [str]  | the folder for log file          | ./Result/       |
| modelDir      |  [str]  | the folder for saving model      | ./Checkpoint/   |
| resultImgDir  |  [str]  | the folder for output            | ./ResultImg/    |
| log           |  [str]  | the folder for tensorboard       | ./log/          |
| sampleNum     |  [int]  | the number of sample for data    | 1               |
| batchSize     |  [int]  | batch size                       | 4               |
| lr            |  [float]| leanring rate                    | 0.001           |
| maxEpochs     |  [int]  | training epoch                   | 30              |
| imgWidth      |  [int]  | the croped width                 | 512             |
| imgHeight     |  [int]  | the croped height                | 256             |
| imgNum        |  [int]  | the number of images for tranin  | 35354           |
| valImgNum     |  [int]  | the number of images for val     | 200             |
| modelName     |  [str]  | the model's name                 | NLCA-Net        |
| dataset       |  [str]  | the dataset's name               | SceneFlow       |


**5) Clean the project (if you want to clean generating files)**
```
$ ./clean.sh
```
---
#### 3. File Structure
```
.
├── Source # source code
│   ├── JackFramework/
|   |   ├── Contrib/
|   |   ├── DatasetReader/
|   |   ├── Evalution/
|   |   ├── NN/
|   |   ├── Proc/
|   |   ├── SysBasic/
|   |   ├── UserTemplate/
|   |   ├── FileHandler/ 
│   |   └── ...
│   ├── setup.py
│   └── ...
├── LICENSE
└── README.md
```

---
#### To do
#### 2021-07-10
1. rewirte the readme;
2. code refacotoring for contrib;
3. refactor the code;

---
#### Update log
##### 2021-07-01
1. Add action for github;
2. Add some information for JackFramework;
3. Write the ReadMe.

##### 2021-05-28
1. Write ReadMe;
2. Add setup.py;
