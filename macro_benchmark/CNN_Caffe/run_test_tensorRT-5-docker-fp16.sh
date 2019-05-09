#!/bin/bash

echo "Running md5 checksum on Caffe models ..."
if md5sum -c checksum.md5; then
	echo "Caffe models checksum pass"
else
	echo "Caffe models checksum fail"
	exit 1
fi

CR_PWD=$(pwd)

mkdir results_infer_trt_fp16
cd results_infer_trt_fp16

for bs in 16 32 64; do
    trtexec --deploy=$CR_PWD/googlenet_bvlc.prototxt  --output=prob --batch=$bs --iterations=1 --avgRuns=500  --model=$CR_PWD/googlenet_bvlc.caffemodel --fp16 | tee googlenet_bvlc_${bs}_fp16.txt
    trtexec --deploy=$CR_PWD/ResNet-50-deploy.prototxt  --output=prob --batch=$bs --iterations=1 --avgRuns=500  --model=$CR_PWD/ResNet-50-model.caffemodel --fp16 | tee resnet50_${bs}_fp16.txt
    trtexec --deploy=$CR_PWD/ResNet-152-deploy.prototxt  --output=prob --batch=$bs --iterations=1 --avgRuns=500  --model=$CR_PWD/ResNet-152-model.caffemodel --fp16 | tee resnet152_${bs}_fp16.txt
    trtexec --deploy=$CR_PWD/densenet121_deploy.prototxt  --output=prob --batch=$bs --iterations=1 --avgRuns=500  --model=$CR_PWD/densenet121.caffemodel --fp16 | tee densenet121_${bs}_fp16.txt
    trtexec --deploy=$CR_PWD/squeezenet_v1.1_deploy.prototxt  --output=prob --batch=$bs --iterations=1 --avgRuns=500  --model=$CR_PWD/squeezenet_v1.1.caffemodel --fp16 | tee squeezenetv1.1_${bs}_fp16.txt
done

python process_results.py --infer_trt --infer_trt_precision=fp16
