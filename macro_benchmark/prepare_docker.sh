#!/bin/bash

sudo docker pull nvcr.io/nvidia/tensorflow:19.09-py3

cd BERT_PyTorch
sudo docker build --pull -t aimatrix:pytorch .
cd ..
