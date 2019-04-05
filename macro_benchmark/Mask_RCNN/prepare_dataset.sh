#!/bin/bash

if [ ! -f "mask_rcnn_coco.h5" ]; then
    wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
fi

if [ ! -f "coco_dataset/train2014.zip" ] || [ ! -f "coco_dataset/val2014.zip" ] || [ ! -f "coco_dataset/annotations_trainval2014.zip" ]; then
	python download_coco.py
fi

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi
