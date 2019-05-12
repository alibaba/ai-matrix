#!/bin/bash

if [ -d results_trt_fp32 ]; then
    mv results_trt_fp32 results_trt_fp32_$(date +%Y%m%d%H%M%S)
fi
mkdir results_trt_fp32

### CNN_Tensorflow ###
if [ -d CNN_Tensorflow/results_infer_trt_fp32 ]; then
	cp -r CNN_Tensorflow/results_infer_trt_fp32  results_trt_fp32/results_cnn_tf
else
	echo "CNN_Tensorflow/results_infer_trt_fp32 does not exist, check if test is ran successfully"
fi

### CNN_Caffe ###
if [ -d CNN_Caffe/results_infer_trt_fp32 ]; then
	cp -r CNN_Caffe/results_infer_trt_fp32  results_trt_fp32/results_cnn_caffe
else
	echo "CNN_Caffe/results_infer_trt_fp32 does not exist, check if test is ran successfully"
fi

### SSD_Caffe ###
if [ -d SSD_Caffe/tensorrt/sampleSSD/results_infer_fp32 ]; then
        cp -r SSD_Caffe/tensorrt/sampleSSD/results_infer_fp32  results_trt_fp32/results_ssd_caffe
else
        echo "SSD_Caffe/tensorrt/sampleSSD/results_infer_fp32 does not exist, check if test is ran successfully"
fi
python process_results.py --results_dir=./results_trt_fp32 --trt_fp32

