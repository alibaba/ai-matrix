#!/bin/bash

#Only run the following command for the first time when you need to download the coco dataset
#python3 coco.py train --dataset=../../coco_dataset --year=2014 --model=../../mask_rcnn_coco.h5 --download=True

python3 coco.py train --dataset=../../coco_dataset --year=2014 --model=../../mask_rcnn_coco.h5