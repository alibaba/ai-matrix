#!/bin/bash

if [ ! -f "ResNet-152-model.caffemodel" ]; then
	wget https://zenodo.org/api/files/aa42ffda-dc69-4110-8a52-32f9dc7092b8/ResNet-152-model.caffemodel
fi

if [ ! -f "ResNet-50-model.caffemodel" ]; then
	wget https://zenodo.org/api/files/aa42ffda-dc69-4110-8a52-32f9dc7092b8/ResNet-50-model.caffemodel
fi

if [ ! -f "densenet121.caffemodel" ]; then
	wget https://zenodo.org/api/files/aa42ffda-dc69-4110-8a52-32f9dc7092b8/densenet121.caffemodel
fi

if [ ! -f "googlenet_bn_stepsize_6400_iter_1200000.caffemodel" ]; then
	wget https://zenodo.org/api/files/aa42ffda-dc69-4110-8a52-32f9dc7092b8/googlenet_bn_stepsize_6400_iter_1200000.caffemodel
fi

if [ ! -f "googlenet_bvlc.caffemodel" ]; then
	wget https://zenodo.org/api/files/aa42ffda-dc69-4110-8a52-32f9dc7092b8/googlenet_bvlc.caffemodel
fi

if [ ! -f "squeezenet_v1.1.caffemodel" ]; then
	wget https://zenodo.org/api/files/aa42ffda-dc69-4110-8a52-32f9dc7092b8/squeezenet_v1.1.caffemodel
fi

echo "Running md5 checksum on Caffe models ..."
if md5sum -c checksum.md5; then
	echo "Caffe models checksum pass"
else
	echo "Caffe models checksum fail"
	exit 1
fi

