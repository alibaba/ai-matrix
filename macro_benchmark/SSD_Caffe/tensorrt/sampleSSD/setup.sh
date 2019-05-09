#!/bin/bash
make
cd ../data

if [ ! -d ssd ]; then
  git clone https://github.com/aimatrix-alibaba/ssd-caffe-trt.git
  mv ssd-caffe-trt/ssd.tar.gz ./
  tar -xzvf ssd.tar.gz
  echo "Running md5 checksum on downloaded dataset ..."
  if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
  else
        echo "Dataset checksum fail"
        exit 1
  fi
fi

cd ../sampleSSD

