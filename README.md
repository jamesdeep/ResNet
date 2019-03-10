Reproduce ResNet using Tensorflow
=====================================
Reproduce of the original ResNet: https://github.com/KaimingHe/deep-residual-networks using tensorflow.

# Requirements
Tensorflow 1.12
CUDA 9.0

# Introduction
This repository reproduce the original Renset v1 on ImageNet using Tensorflow low level API. I write all the model, training and data pipeline code from scratch. The training code utilizes multi-GPU and multi-processes and is really fast. The training of ResNet-50 on Imagenet can be completed within 4 days of 120 epoches on a 4-GPUs Titan XP with batch size 256.

## Trained models
The trained ResNet models achieve better error rates than the [original ResNet-v1 models](https://github.com/KaimingHe/deep-residual-networks) and a litter better than another [reproduce project](https://github.com/tornadomeet/ResNet) which uses ResNetv2.

### ImageNet 1K

Imagenet 1000 class dataset with 1.2 million images.

single center crop (224x224) validation error rate(%)

| Network       | Top-1 error | Top-5 error | Traind Model |
| :------------ | :---------: | :---------: | :-------------: |
| ResNet-50     | 24.36       | 7.46        | realease later |

### Training Log
All the training detail you can found [here](./log), which include lr schedular, batch-size etc, and you can also see the training speed with the corresponding logs. 

### Notes
1. Data augmentation has critical impacts on the accuracy. The data augmentation is the same with the original ResNet paper except color augmentation. I reference the color augmentation from a [official reproduce of Tensorflow](https://github.com/tensorflow/benchmarks/blob/master/scripts/tf_cnn_benchmarks/preprocessing.py). This is why our accuracy is higher than the original paper. Also I note that [another reproduce](https://github.com/facebook/fb.resnet.torch) use different scale and ratio augmentation which aslo achieves better accuracy than the original one. Also be aware that too much data augmentation may have negative impacts on the accuracy. I use python-opencv2 module to handle the data augmentation. Notice that set the threads to 1 after import cv2 by using 
```shell
cv2.setNumThreads(1)
```
, because cv2 uses multithreads to handle image processing, which has negative effects in the multi-processes environment and slows down the training speed.

2. Learning rate schedule: start at 0.1, and decrease by 10 at epoch 30, 60 ,90. The training ends at the 120th epoch.

3. Batch norm layer: In the tensorflow implemetation, use[tf.nn.fused_batch_norm](https://www.tensorflow.org/api_docs/python/tf/nn/fused_batch_norm) rather than [tf.nn.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization) or [tf.layers.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization), because it is much faster. Just like the [original ResNet-v1 models](https://github.com/KaimingHe/deep-residual-networks) does, the provided mean and variance are strictly computed using average (not moving average) on a few training batch(25 batches, in facts, I also try 1000 batches, and the accuracy is almost the same) after the training procedure.

4.Batch size: 256, 4 GPUs. Each GPU handles 64.

5.Records file: since the Imagenet is large, we converts the original Imagenet imgs to records, just like tensorlow official and mxnet do. The training images are randomly saved to 1024 records, the validation images are randomly saved to 128 records.

6. Multi-processes: since python multi-threads does not utilize multi-CPU due to GIL, we have to use multi-process module to handle the computation intensive data augmentation task. This is vital to maximize the training speed. I use ZMQ IPC protocol to handle the communication among processes, which reference to [this](https://tensorpack.readthedocs.io/tutorial/efficient-dataflow.html).

7. I do not substracting pixel mean of input images.
