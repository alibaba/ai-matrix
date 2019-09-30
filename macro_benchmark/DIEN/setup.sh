#!/bin/bash

if [ ! -f "data.tar.gz" ]; then
	wget https://zenodo.org/api/files/d63bab5d-e8ce-4ec8-8dbb-1656d4f653ec/data.tar.gz
fi

if [ ! -f "data1.tar.gz" ]; then
	wget https://zenodo.org/api/files/d63bab5d-e8ce-4ec8-8dbb-1656d4f653ec/data1.tar.gz
fi

if [ ! -f "data2.tar.gz" ]; then
	wget https://zenodo.org/api/files/d63bab5d-e8ce-4ec8-8dbb-1656d4f653ec/data2.tar.gz
fi

if [ ! -f "dnn_best_model_trained/ckpt_noshuffDIEN3.data-00000-of-00001" ]; then
	wget https://zenodo.org/api/files/d63bab5d-e8ce-4ec8-8dbb-1656d4f653ec/ckpt_noshuffDIEN3.data-00000-of-00001
fi

echo "Running md5 checksum on Caffe models ..."
if md5sum -c checksum.md5; then
	echo "Caffe models checksum pass"
else
	echo "Caffe models checksum fail"
	exit 1
fi

