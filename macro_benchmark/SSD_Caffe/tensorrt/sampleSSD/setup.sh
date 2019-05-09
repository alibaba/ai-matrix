#!/bin/bash

cd ..

if [ ! -d data ]; then
    mkdir data
fi

cd data

git clone https://github.com/aimatrix-alibaba/ssd-caffe-trt.git
mv ssd-caffe-trt/ssd.tar.gz ./
tar -xzvf ssd.tar.gz
cd ../sampleSSD

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi
