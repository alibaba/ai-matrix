#!/bin/bash

export PYTHONPATH=`pwd`/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

if [ ! -f "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" ]; then
    cd models/VGGNet
    wget https://zenodo.org/record/3463718/files/VGG_ILSVRC_16_layers_fc_reduced.caffemodel
    cd ../..
fi

if [ ! -f "models/VGGNet/VOC0712_pretrained/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel" ]; then
    cd models/VGGNet/VOC0712_pretrained/SSD_300x300
    wget https://zenodo.org/record/3463718/files/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel
    cd ../../../..
fi

cd data

if [ ! -f "VOCtrainval_06-Nov-2007.tar" ]; then
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    tar -xvf VOCtrainval_06-Nov-2007.tar
fi

if [ ! -f "VOCtest_06-Nov-2007.tar" ]; then
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    tar -xvf VOCtest_06-Nov-2007.tar
fi

if [ ! -f "VOCtrainval_11-May-2012.tar" ]; then
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    tar -xvf VOCtrainval_11-May-2012.tar
fi

cd ..
./data/VOC0712/create_list.sh
./data/VOC0712/create_data.sh

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi
