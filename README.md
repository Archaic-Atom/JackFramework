>This is a trainin framework based on pythorch, if you have any question please send an e-mail to raoxi36@foxmail.com

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
**1) Install the JackFramework lib**
```
$ ./install.sh
```
**2) Check the version (optional)**
```
$ python -c "import JackFramework as jf; print(jf.version())"
```

**3) Clean the project**
```
$ ./clean.sh
```

#### 3. File Structure
```
.
├── Source # source code
│   ├── JackFramework/
│   ├── setup.py
│   └── ...
├── LICENSE
└── README.md
```

---
#### Update log

##### 2021-05-28
1. Write ReadMe
2. Add setup.py