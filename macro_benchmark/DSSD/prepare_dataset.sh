#!/bin/bash

if [ ! -f "models/ResNet-101/ResNet-101-model.caffemodel" ]; then
    if [ ! -d "models" ]; then
	mkdir models
    fi
    cd models
    if [ ! -d "ResNet-101" ]; then
	mkdir ResNet-101
    fi
    cd ResNet-101
    wget https://zenodo.org/record/3463685/files/ResNet-101-model.caffemodel
    cd ../..
fi

if [ ! -f "models/ResNet-18/resnet-18.caffemodel" ]; then
    if [ ! -d "models" ]; then
        mkdir models
    fi
    cd models
    if [ ! -d "ResNet-18" ]; then
        mkdir ResNet-18
    fi
    cd ResNet-18
    wget https://zenodo.org/record/3463685/files/resnet-18.caffemodel
    cd ../..
fi

export PYTHONPATH=`pwd`/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

if [ ! -d "VOC0712" ]; then
    mkdir VOC0712
    cd VOC0712
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

    tar -xvf VOCtrainval_06-Nov-2007.tar
    tar -xvf VOCtest_06-Nov-2007.tar
    tar -xvf VOCtrainval_11-May-2012.tar

    cd ..
else
    echo "VOC0712 already exists"
fi

./data/VOC0712/create_list.sh
./data/VOC0712/create_data.sh

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi
