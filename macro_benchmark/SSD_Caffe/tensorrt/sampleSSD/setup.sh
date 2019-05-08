#!/bin/bash

cd ../data
git clone https://github.com/aimatrix-alibaba/ssd-caffe-trt.git
mv ssd-caffe-trt/ssd.tar.gz ./
tar -xzvf ssd.tar.gz
cd ../sampleSSD
