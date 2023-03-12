# Source Label Adaptation
The official Pytorch implementation of "Semi-Supervised Domain Adaptation with Source Label Adaptation" accepted by CVPR2023. Check more details of this work in our paper: [[Arxiv]](https://arxiv.org/abs/2302.02335).

## Python version & Packages

`python==3.8.13`

```
configargparse==1.5.3
torch==1.12.0
torchvision==0.13.0
tensorboard==2.9.0
Pillow==9.0.1
numpy==1.22.3
```

## Usage

1. Dataset Preparement
    
    Following [MME](https://github.com/VisionLearningGroup/SSDA_MME) to download the dataset and the split files.

    In `config.yaml`, specify the path for the dataset, and the path for the split files.
    - all: the file with all samples.
    - 1shot:
        - train: training split for 1-shot setting.
        - test: test split for 1-shot setting.
    - 3shot:
        - train: training split for 3-shot setting.
        - test: test split for 3-shot setting.
    - val: validation split.

2. Running

Take MME + SLA on 3-shot A -> C Office-Home dataset as example:

```
python --method MME_LC --source 0 --target 1 --seed 1102 --num_iters 10000 --shot 3shot --alpha 0.3 --update_interval 500 --warmup 500 --T 0.6
```

## Acknowledgement

This code is partially based on [MME](https://github.com/VisionLearningGroup/SSDA_MME), and [DeepDA](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA).
