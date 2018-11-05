#!/bin/bash

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

### CNN_Tensorflow ###
if [ -f CNN_Tensorflow/results_train/results.csv ]; then
	cp CNN_Tensorflow/results_train/results.csv results/results_cnn_train.csv
else
	echo "CNN_Tensorflow/results_train/results.csv does not exit, check if test is ran successfully"
fi

if [ -f CNN_Tensorflow/results_infer/results.csv ]; then
	cp CNN_Tensorflow/results_infer/results.csv results/results_cnn_infer.csv
else
	echo "CNN_Tensorflow/results_infer/results.csv does not exit, check if test is ran successfully"
fi

### DeepInterest ###
if [ -f DeepInterest/results/results_train.csv ]; then
	cp DeepInterest/results/results_train.csv results/results_deepinterest_train.csv
else
	echo "DeepInterest/results/results_train.csv does not exit, check if test is ran successfully"
fi

if [ -f DeepInterest/results/results_infer.csv ]; then
	cp DeepInterest/results/results_infer.csv results/results_deepinterest_infer.csv
else
	echo "DeepInterest/results/results_infer.csv does not exit, check if test is ran successfully"
fi

### Mask_RCNN ###
if [ -f Mask_RCNN/results/results_train.csv ]; then
	cp Mask_RCNN/results/results_train.csv results/results_maskrcnn_train.csv
else
	echo "Mask_RCNN/results/results_train.csv does not exit, check if test is ran successfully"
fi

if [ -f Mask_RCNN/results/results_infer.csv ]; then
	cp Mask_RCNN/results/results_infer.csv results/results_maskrcnn_infer.csv
else
	echo "Mask_RCNN/results/results_infer.csv does not exit, check if test is ran successfully"
fi

### NMT ###
if [ -f NMT/results/results_train.csv ]; then
	cp NMT/results/results_train.csv results/results_nmt_train.csv
else
	echo "NMT/results/results_train.csv does not exit, check if test is ran successfully"
fi

if [ -f NMT/results/results_infer.csv ]; then
	cp NMT/results/results_infer.csv results/results_nmt_infer.csv
else
	echo "NMT/results/results_infer.csv does not exit, check if test is ran successfully"
fi

### SSD ###
if [ -f SSD/results/results_train.csv ]; then
	cp SSD/results/results_train.csv results/results_ssd_train.csv
else
	echo "SSD/results/results_train.csv does not exit, check if test is ran successfully"
fi

if [ -f SSD/results/results_infer.csv ]; then
	cp SSD/results/results_infer.csv results/results_ssd_infer.csv
else
	echo "SSD/results/results_infer.csv does not exit, check if test is ran successfully"
fi

python process_results.py
