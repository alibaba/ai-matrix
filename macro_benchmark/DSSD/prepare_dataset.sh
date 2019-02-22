#!/bin/bash

export PYTHONPATH=`pwd`/python

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
    ./data/VOC0712/create_list.sh
    ./data/VOC0712/create_data.sh
else
    echo "VOC0712 already exists"
fi

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi
