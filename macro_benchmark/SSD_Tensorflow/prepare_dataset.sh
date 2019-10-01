#!/bin/bash

if [ ! -f "checkpoints/ssd_300_vgg.ckpt.zip" ]; then
        cd checkpoints
    wget https://zenodo.org/api/files/9cc1a8c5-c46c-425d-9fce-05795e3c25fd/ssd_300_vgg.ckpt.zip
    cd ..
fi

cd checkpoints
if [ ! -f "ssd_300_vgg.ckpt.data-00000-of-00001" ] || [ ! -f "ssd_300_vgg.ckpt.index" ]; then
    unzip ssd_300_vgg.ckpt.zip
else
    echo "Checkpoint already unzipped"
fi
cd ..

if [ ! -d "VOC2007" ]; then
    mkdir VOC2007
    cd VOC2007
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    tar -xvf VOCtrainval_06-Nov-2007.tar
    mv VOCdevkit/VOC2007 trainval
    tar -xvf VOCtest_06-Nov-2007.tar
    mv VOCdevkit/VOC2007 test
    rm -r VOCdevkit
    mkdir trainval_tf test_tf
    cd ..
    python tf_convert_data.py --dataset_name=pascalvoc --dataset_dir=./VOC2007/trainval/ --output_name=voc_2007_train --output_dir=./VOC2007/trainval_tf
    python tf_convert_data.py --dataset_name=pascalvoc --dataset_dir=./VOC2007/test/ --output_name=voc_2007_test --output_dir=./VOC2007/test_tf
else
    echo "VOC2007 already exists"
fi

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi

