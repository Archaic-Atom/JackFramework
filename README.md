>This is a training framework based on pythorch, if you have any question please send an e-mail to raoxi36@foxmail.com

#### 0. Software Environment
**1) OS Environment**
os >= linux 16.04

cudaToolKit == 10.1

cudnn == 7.3.6

**2) Python Environment**
python == 3.8.5

pythorch >= 1.15.0

numpy == 1.14.5

opencv == 3.4.0

PIL == 5.1.0

#### 1. Hardware Environment
This framework is only used in GPUs.

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

**3) the template for using the JackFramework**

you can find the template project in: https://github.com/Archaic-Atom/FameworkTemplate

**4) Clean the project**
```
$ ./clean.sh
```

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
#### Update log
##### 2021-07-01
1. Add action for github;
2. Add some information for JackFramework;
3. Write the ReadMe.

##### 2021-05-28
1. Write ReadMe;
2. Add setup.py;
