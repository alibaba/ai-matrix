#!/bin/bash

echo "Running md5 checksum on Caffe models ..."
if md5sum -c checksum.md5; then
	echo "Caffe models checksum pass"
else
	echo "Caffe models checksum fail"
	exit 1
fi

CR_PWD=$(pwd)
DEVICE=1
DT=$1

mkdir results
cd results

echo "=============Inference on googlenet============="
for bs in 16 32 64; do
      if [ "$DT" == "fp32" ]; then
          giexec --deploy=$CR_PWD/googlenet_bvlc.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/googlenet_bvlc.caffemodel | tee googlenet_bvlc_${bs}_fp32.txt
      elif [ "$DT" == "fp16" ]; then
          giexec --deploy=$CR_PWD/googlenet_bvlc.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/googlenet_bvlc.caffemodel --fp16 | tee googlenet_bvlc_${bs}_fp16.txt
      elif [ "$DT" == "int8" ]; then
          giexec --deploy=$CR_PWD/googlenet_bvlc.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/googlenet_bvlc.caffemodel --int8 | tee googlenet_bvlc_${bs}_int8.txt
      else 
          exit 2
      fi
done


echo "=============Inference on resnet50============="
for bs in 16 32 64; do
      if [ "$DT" == "fp32" ]; then
          giexec --deploy=$CR_PWD/ResNet-50-deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/ResNet-50-model.caffemodel | tee resnet50_${bs}_fp32.txt
      elif [ "$DT" == "fp16" ]; then
          giexec --deploy=$CR_PWD/ResNet-50-deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/ResNet-50-model.caffemodel --fp16 | tee resnet50_${bs}_fp16.txt
      elif [ "$DT" == "int8" ]; then
          giexec --deploy=$CR_PWD/ResNet-50-deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/ResNet-50-model.caffemodel --int8 | tee resnet50_${bs}_int8.txt
      else 
          exit 2
      fi
done

echo "=============Inference on resnet152============="
for bs in 16 32 64; do
      if [ "$DT" == "fp32" ]; then
          giexec --deploy=$CR_PWD/ResNet-152-deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/ResNet-152-model.caffemodel | tee resnet152_${bs}_fp32.txt
      elif [ "$DT" == "fp16" ]; then
          giexec --deploy=$CR_PWD/ResNet-152-deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/ResNet-152-model.caffemodel --fp16 | tee resnet152_${bs}_fp16.txt
      elif [ "$DT" == "int8" ]; then
          giexec --deploy=$CR_PWD/ResNet-152-deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/ResNet-152-model.caffemodel --int8 | tee resnet152_${bs}_int8.txt
      else 
          exit 2
      fi
done

echo "=============Inference on densenet121============="
for bs in 16 32 64; do
      if [ "$DT" == "fp32" ]; then
          giexec --deploy=$CR_PWD/densenet121_deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/densenet121.caffemodel | tee densenet121_${bs}_fp32.txt
      elif [ "$DT" == "fp16" ]; then
          giexec --deploy=$CR_PWD/densenet121_deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/densenet121.caffemodel --fp16 | tee densenet121_${bs}_fp16.txt
      elif [ "$DT" == "int8" ]; then
          giexec --deploy=$CR_PWD/densenet121_deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/densenet121.caffemodel --int8 | tee densenet121_${bs}_int8.txt
      else 
          exit 2
      fi
done

echo "=============Inference on squeezenet v1.1============="
for bs in 16 32 64; do
      if [ "$DT" == "fp32" ]; then
          giexec --deploy=$CR_PWD/squeezenet_v1.1_deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/squeezenet_v1.1.caffemodel | tee squeezenetv1.1_${bs}_fp32.txt
      elif [ "$DT" == "fp16" ]; then
          giexec --deploy=$CR_PWD/squeezenet_v1.1_deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/squeezenet_v1.1.caffemodel --fp16 | tee squeezenetv1.1_${bs}_fp16.txt
      elif [ "$DT" == "int8" ]; then
          giexec --deploy=$CR_PWD/squeezenet_v1.1_deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/squeezenet_v1.1.caffemodel --int8 | tee squeezenetv1.1_${bs}_int8.txt
      else 
          exit 2
      fi
done
