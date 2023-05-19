<div align="center">

# LaneATT

Reference: https://github.com/lucastabelini/LaneATT

### Table of contents
- [LaneATT](#laneatt)
    - [Table of contents](#table-of-contents)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Install](#2-install)
    - [3. Getting started](#3-getting-started)
      - [Datasets](#datasets)
- [train \& validation data (~10 GB)](#train--validation-data-10-gb)
- [test images (~10 GB)](#test-images-10-gb)
- [test annotations](#test-annotations)
      - [Training \& testing](#training--testing)
    - [4. Results](#4-results)
      - [TuSimple](#tusimple)


### 1. Prerequisites
- Python >= 3.5
- PyTorch == 1.6, tested on CUDA 10.2. The models were trained and evaluated on PyTorch 1.6. When testing with other versions, the results (metrics) are slightly different.
- CUDA, to compile the NMS code
- Other dependencies described in `requirements.txt`

The versions described here were the lowest the code was tested with. Therefore, it may also work in other earlier versions, but it is not guaranteed (e.g., the code might run, but with different outputs).

### 2. Install
Conda is not necessary for the installation, as you can see, I only use it for PyTorch and Torchvision.
Nevertheless, the installation process here is described using it.

```bash
conda create -n laneatt python=3.8 -y
conda activate laneatt
conda install pytorch==1.6 torchvision -c pytorch
pip install -r requirements.txt
cd lib/nms; python setup.py install; cd -
```

### 3. Getting started
#### Datasets

We used TuSimple in our experiments
Firstly move to directory where this repository is cloned. 
'''bash
mkdir datasets # if it does not already exists
cd datasets
# train & validation data (~10 GB)
mkdir tusimple
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip"
unzip train_set.zip -d tusimple
# test images (~10 GB)
mkdir tusimple-test
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip"
unzip test_set.zip -d tusimple-test
# test annotations
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json" -P tusimple-test/
cd ..

'''
#### Training & testing
Train a model:

```
python main.py train --exp_name example --cfg example.yml
```
The configuration files are located in cfgs folder.

After running this command, a directory `experiments` should be created (if it does not already exists). Another
directory `laneatt_r34_tusimple` will be inside it, containing data related to that experiment (e.g., model checkpoints, logs, evaluation results, etc)

Evaluate a model:
```
python main.py test --exp_name example
```

This command will evaluate the model saved in the last checkpoint of the experiment `example` (inside `experiments`).
If you want to evaluate another checkpoint, the `--epoch` flag can be used. For other flags, please see `python main.py -h`. To **visualize the predictions**, run the above command with the additional flag `--view all`.


### 4. Results

The weights for each of the models can be downloaded from here:
To check results of original resnet models, please download the experiment files from original authors using following commands.
'''bash
gdown "https://drive.google.com/uc?id=1R638ou1AMncTCRvrkQY6I-11CPwZy23T"
unzip laneatt_experiments.zip
'''

Weights of mobileone, mobilenet and shufflenet can be downloaded from following link: https://drive.google.com/drive/folders/1KD3xFiFodNZwR5VsQByieXAkSaExVmqJ?usp=sharing
#### TuSimple

| Backbones | Accuracy% | FP | FN | FPS |
| --------- | --------- | -- | -- | --- |
| ResNet18 | 94.99 | 0.0975 | 0.0364 | 125.8841 |
| ResNet34 | 94.71 | 0.1279 | 0.0403 | 106.8609 |
| ResNet122 | 95.36 | 0.1182 | 0.0356 | 31.8189 |
| MobileOne | 95.33 | 0.0611 | 0.0300 | 47.7609 |
| ShuffleNet | 95.20 | 0.0738 | 0.0364 | 93.1477 |
| Mobilenet | 95.18 | 0.0547 | 0.0360 | 85.95 |

Qualitative Results: https://drive.google.com/file/d/1BG4OzQGubZ5yYiItCAl79LWFpXBy9Uu7/view?usp=sharing