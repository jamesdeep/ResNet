Reproduce ResNet using Tensorflow
=====================================
# ResNet
Reproduce of the original ResNet: https://github.com/KaimingHe/deep-residual-networks

# Requirements
Tensorflow 1.12
CUDA 9.0

# Introduction
This repository reproduce the original Renset on ImageNet using Tensorflow low level API. I write all the model, training and data pipeline code from scratch. The training code utilizes multi-GPU and multi-processes and is really fast. The training of ResNet-50 on Imagenet can be completed within 4 days of 120 epoches on a 4-GPU Titan XP with batch size 256.

## Trained models
The trained ResNet models achieve better error rates than the [original ResNet-v1 models](https://github.com/KaimingHe/deep-residual-networks).

### ImageNet 1K

Imagenet 1000 class dataset with 1.2 million images.

single center crop (224x224) validation error rate(%)

| Network       | Top-1 error | Top-5 error | Traind Model |
| :------------ | :---------: | :---------: | :-------------: |
| ResNet-50     | 24.36       | 7.46        | realease later |

### Training Log
All the training detail you can found [here](./log), which include lr schedular, batch-size etc, and you can also see the training speed with the corresponding logs. 

### Notes
