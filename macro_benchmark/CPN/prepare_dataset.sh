#!/bin/bash

if [ ! -d "data/imagenet_weights" ]; then
	mkdir data/imagenet_weights
	cd data/imagenet_weights
	wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
	tar -xvzf resnet_v1_101_2016_08_28.tar.gz
	cd -
fi

if [ ! -f "data/COCO/person_detection_minival411_human553.json" ]; then
	cd data/COCO
	gdown https://drive.google.com/uc?id=1RHWE0xYcnDyUiB5pHKfJZYJsOeN_rr7A
	mv person_detection_minival411_human553.json.coco person_detection_minival411_human553.json
	cd -
fi

if [ ! -f "data/COCO/person_keypoints_minival2014.json" ]; then
	cd data/COCO
	gdown https://drive.google.com/uc?id=1Y7_33TdwPV_0lhou-PhXxb_91uoYUcKP
	cd -
fi

if [ ! -f "data/COCO/person_keypoints_trainvalminusminival2014.json" ]; then
	cd data/COCO
	gdown https://drive.google.com/uc?id=1mFg_hC7FN9rSv-H6VA3JfDGfL6pmeIaW
	cd -
fi

if [ ! -f "../Mask_RCNN/coco_dataset/train2014.zip" ] || [ ! -f "../Mask_RCNN/coco_dataset/val2014.zip" ] || [ ! -f "../Mask_RCNN/coco_dataset/annotations_trainval2014.zip" ]; then
	cd ../Mask_RCNN
	python download_coco.py
	cd -
fi

if [ ! -d "data/COCO/MSCOCO" ]; then
	ln -s ../../../Mask_RCNN/coco_dataset data/COCO/MSCOCO
fi

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi
