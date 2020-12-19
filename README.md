# recycle_detection : inference code
**AI Grand Challenge (Nov 16, 2020 ~ Nov 20,2020)**

## Environments
```
Ubuntu 18.04.5 LTS   
Python 3.7  
CUDA 10.2  
```

## Install
```
$ git clone https://github.com/SeaRealE/recycle_detection.git
$ cd recycle_detection
$ pip install -r requirements.txt
$ pip install mmpycocotools
$ pip install mmcv-full==1.1.6+torch1.5.0+cu102 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html --use-deprecated=legacy-resolver
```

## Run
⚠️ **you need a file** `weights.pth`
```
$ python predict.py {FILE_PATH}      
```
e.g. `$ python predict.py ./test/`