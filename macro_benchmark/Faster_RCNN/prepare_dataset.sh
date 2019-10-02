#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

if [ ! -f "pretrained_model/vgg16_faster_rcnn_iter_70000.caffemodel" ]; then
    if [ ! -d "pretrained_model" ]; then
        mkdir pretrained_model
    fi
    cd pretrained_model
        wget https://zenodo.org/record/3463689/files/vgg16_faster_rcnn_iter_70000.caffemodel
        cd ..
fi

if [ ! -d "VOCdevkit" ]; then
    if [ ! -f "../DSSD/VOC0712/VOCtrainval_06-Nov-2007.tar" ]; then
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
        tar -xvf VOCtrainval_06-Nov-2007.tar
    else
        tar -xvf ../DSSD/VOC0712/VOCtrainval_06-Nov-2007.tar
    fi

    if [ ! -f "../DSSD/VOC0712/VOCtest_06-Nov-2007.tar" ]; then
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
        tar -xvf VOCtest_06-Nov-2007.tar
    else
        tar -xvf ../DSSD/VOC0712/VOCtest_06-Nov-2007.tar
    fi

    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
    tar -xvf VOCdevkit_08-Jun-2007.tar
else
    echo "VOCdevkit already exists"
fi

cd data
ln -s ../VOCdevkit VOCdevkit2007
cd ..

./data/scripts/fetch_imagenet_models.sh

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi
