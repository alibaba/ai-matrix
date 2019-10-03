#!/bin/bash
#git clone https://github.com/aimatrix-alibaba/aimatrix-pretrained-weights.git

if [ ! -d "aimatrix-pretrained-weights" ]; then
	mkdir aimatrix-pretrained-weights
fi

cd aimatrix-pretrained-weights

if [ ! -d "CNN_Tensorflow" ]; then
	mkdir CNN_Tensorflow
fi

cd CNN_Tensorflow

if [ ! -f "imagenet_validation_TF.tgz" ]; then
	wget https://zenodo.org/record/3470355/files/imagenet_validation_TF.tgz
fi

if [ ! -f "log_densenet121.tgz" ]; then
	wget https://zenodo.org/record/3470355/files/log_densenet121.tgz
fi

if [ ! -f "log_googlenet.tgz" ]; then
	wget https://zenodo.org/record/3470355/files/log_googlenet.tgz
fi

if [! -f "log_resnet152.tgz" ]; then
	wget https://zenodo.org/record/3470355/files/log_resnet152.tgz
fi

if [ ! -f "log_resnet50.tgz" ]; then
	wget https://zenodo.org/record/3470355/files/log_resnet50.tgz
fi

tar -xzvf imagenet_validation_TF.tgz
tar -xzvf log_googlenet.tgz
tar -xzvf log_resnet50.tgz
tar -xzvf log_resnet152.tgz
tar -xzvf log_densenet121.tgz

if [ ! -d "graphs_NCHW" ]; then
	mkdir graphs_NCHW
fi

cd graphs_NCHW

if [ ! -f "frozen_graph_googlenet_fp32_32.pb" ]; then
	wget https://zenodo.org/record/3470662/files/frozen_graph_googlenet_fp32_32.pb
fi

if [ ! -f "frozen_graph_resnet152_fp32_32.pb" ]; then
	wget https://zenodo.org/record/3470662/files/frozen_graph_resnet152_fp32_32.pb
fi

if [ ! -f "frozen_graph_resnet50_fp32_32.pb" ]; then
	wget https://zenodo.org/record/3470662/files/frozen_graph_resnet50_fp32_32.pb
fi

cd ..

if [ ! -d "graphs_NHWC" ]; then
        mkdir graphs_NHWC
fi

cd graphs_NHWC

if [ ! -f "frozen_graph_densenet121_fp32_32.pb" ]; then
	wget https://zenodo.org/record/3470666/files/frozen_graph_densenet121_fp32_32.pb
fi

if [ ! -f "frozen_graph_googlenet_fp32_32.pb" ]; then
	wget https://zenodo.org/record/3470666/files/frozen_graph_googlenet_fp32_32.pb
fi

if [ ! -f "frozen_graph_resnet152_fp32_32.pb" ]; then
	wget https://zenodo.org/record/3470666/files/frozen_graph_resnet152_fp32_32.pb
fi

if [ ! -f "frozen_graph_resnet50_fp32_32.pb" ]; then
	wget https://zenodo.org/record/3470666/files/frozen_graph_resnet50_fp32_32.pb
fi

cd ../../..

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi

