#!/bin/bash

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

### CNN_Tensorflow ###
if [ -f CNN_Tensorflow/results_train/results.csv ]; then
	cp CNN_Tensorflow/results_train/results.csv results/results_cnn_train.csv
else
	echo "CNN_Tensorflow/results_train/results.csv does not exist, check if test is ran successfully"
fi

if [ -f CNN_Tensorflow/results_infer/results.csv ]; then
	cp CNN_Tensorflow/results_infer/results.csv results/results_cnn_infer.csv
else
	echo "CNN_Tensorflow/results_infer/results.csv does not exist, check if test is ran successfully"
fi

### DeepInterest ###
if [ -f DeepInterest/results/results_train.csv ]; then
	cp DeepInterest/results/results_train.csv results/results_deepinterest_train.csv
else
	echo "DeepInterest/results/results_train.csv does not exist, check if test is ran successfully"
fi

if [ -f DeepInterest/results/results_infer.csv ]; then
	cp DeepInterest/results/results_infer.csv results/results_deepinterest_infer.csv
else
	echo "DeepInterest/results/results_infer.csv does not exist, check if test is ran successfully"
fi

### Mask_RCNN ###
if [ -f Mask_RCNN/results/results_train.csv ]; then
	cp Mask_RCNN/results/results_train.csv results/results_maskrcnn_train.csv
else
	echo "Mask_RCNN/results/results_train.csv does not exist, check if test is ran successfully"
fi

if [ -f Mask_RCNN/results/results_infer.csv ]; then
	cp Mask_RCNN/results/results_infer.csv results/results_maskrcnn_infer.csv
else
	echo "Mask_RCNN/results/results_infer.csv does not exist, check if test is ran successfully"
fi

### NMT ###
if [ -f NMT/results/results_train.csv ]; then
	cp NMT/results/results_train.csv results/results_nmt_train.csv
else
	echo "NMT/results/results_train.csv does not exist, check if test is ran successfully"
fi

if [ -f NMT/results/results_infer.csv ]; then
	cp NMT/results/results_infer.csv results/results_nmt_infer.csv
else
	echo "NMT/results/results_infer.csv does not exist, check if test is ran successfully"
fi

### SSD_Caffe ###
if [ -f SSD_Caffe/results/results_train.csv ]; then
	cp SSD_Caffe/results/results_train.csv results/results_ssd_train.csv
else
	echo "SSD_Caffe/results/results_train.csv does not exist, check if test is ran successfully"
fi

if [ -f SSD_Caffe/results/results_infer.csv ]; then
	cp SSD_Caffe/results/results_infer.csv results/results_ssd_infer.csv
else
	echo "SSD_Caffe/results/results_infer.csv does not exist, check if test is ran successfully"
fi

### DSSD ###
if [ -f DSSD/results/results_train.csv ]; then
	cp DSSD/results/results_train.csv results/results_dssd_train.csv
else
	echo "DSSD/results/results_train.csv does not exist, check if test is ran successfully"
fi

if [ -f DSSD/results/results_infer.csv ]; then
	cp DSSD/results/results_infer.csv results/results_dssd_infer.csv
else
	echo "DSSD/results/results_infer.csv does not exist, check if test is ran successfully"
fi

### NCF ###
if [ -f NCF/results/results_train.csv ]; then
	cp NCF/results/results_train.csv results/results_ncf_train.csv
else
	echo "DSSD/results/results_train.csv does not exist, check if test is ran successfully"
fi

if [ -f NCF/results/results_infer.csv ]; then
	cp NCF/results/results_infer.csv results/results_ncf_infer.csv
else
	echo "DSSD/results/results_infer.csv does not exist, check if test is ran successfully"
fi

### DIEN ###
if [ -f DIEN/results/results_train.csv ]; then
	cp DIEN/results/results_train.csv results/results_dien_train.csv
else
	echo "DIEN/results/results_train.csv does not exist, check if test is ran successfully"
fi

if [ -f DIEN/results/results_infer.csv ]; then
	cp DIEN/results/results_infer.csv results/results_dien_infer.csv
else
	echo "DIEN/results/results_infer.csv does not exist, check if test is ran successfully"
fi

python process_results.py
