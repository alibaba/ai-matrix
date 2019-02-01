#!/bin/bash
git clone https://github.com/aimatrix-alibaba/aimatrix-pretrained-weights.git

cd aimatrix-pretrained-weights/CNN_Tensorflow
tar -xzvf imagenet_validation_TF.tgz
tar -xzvf log_googlenetv1.tgz
tar -xzvf log_resnet50.tgz
tar -xzvf log_resnet152.tgz
tar -xzvf log_densenet121.tgz
cd ..

cd DeepInterest
unzip checkpoint.zip

cd NCF
tar -xzvf ckpt_512.tgz
