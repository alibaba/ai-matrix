#!/bin/bash

sudo docker pull nvcr.io/nvidia/tensorflow:19.09-py3

cd SSD_ResNet34_PyTorch
sudo docker build --pull -t aimatrix:pytorch .
cd ..
