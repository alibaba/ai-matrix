#!/bin/bash
pc=fp16

if [ -d results_trt_${pc} ]; then
    mv results_trt_${pc} results_trt_${pc}_$(date +%Y%m%d%H%M%S)
fi
mkdir results_trt_${pc}

### CNN_Tensorflow ###
if [ -d CNN_Tensorflow/results_infer_trt_${pc} ]; then
	cp -r CNN_Tensorflow/results_infer_trt_${pc}  results_trt_${pc}/results_cnn_tf
else
	echo "CNN_Tensorflow/results_infer_trt_fp16 does not exist, check if test is ran successfully"
fi

### CNN_Caffe ###
if [ -d CNN_Caffe/results_infer_trt_${pc} ]; then
	cp -r CNN_Caffe/results_infer_trt_${pc}  results_trt_${pc}/results_cnn_caffe
else
	echo "CNN_Caffe/results_infer_trt_fp16 does not exist, check if test is ran successfully"
fi

### SSD_Caffe ###
if [ -d SSD_Caffe/tensorrt/sampleSSD/results_infer_${pc} ]; then
        cp -r SSD_Caffe/tensorrt/sampleSSD/results_infer_${pc}  results_trt_${pc}/results_ssd_caffe
else
        echo "SSD_Caffe/tensorrt/sampleSSD/results_infer_fp16 does not exist, check if test is ran successfully"
fi
python process_results.py --results_dir=./results_trt_${pc} --trt_${pc}

