#!/bin/bash

if [ ! -f "data.tar.gz" ]; then
        wget https://zenodo.org/record/3463683/files/data.tar.gz
fi

if [ ! -f "data1.tar.gz" ]; then
        wget https://zenodo.org/record/3463683/files/data1.tar.gz
fi

if [ ! -f "data2.tar.gz" ]; then
        wget https://zenodo.org/record/3463683/files/data2.tar.gz
fi

if [ ! -f "dnn_best_model_trained/ckpt_noshuffDIEN3.data-00000-of-00001" ]; then
        cd dnn_best_model_trained
        wget https://zenodo.org/record/3463683/files/ckpt_noshuffDIEN3.data-00000-of-00001
        cd ..
fi

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi

tar -jxvf data.tar.gz
mv data/* .
tar -jxvf data1.tar.gz
mv data1/* .
tar -jxvf data2.tar.gz
mv data2/* .
