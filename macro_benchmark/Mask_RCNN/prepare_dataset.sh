#!/bin/bash

if [ ! -f "mask_rcnn_coco.h5" ]; then
    wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
fi

if [ ! -d "coco_dataset" ]; then
    python download_coco.py
fi

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi

