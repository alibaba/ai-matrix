#!/bin/bash

mkdir results_infer_trt_accuracy 

LD_LIBRARY_PATH=./Model/opencv-3.4.0/build/lib/:$LD_LIBRARY_PATH ./bin/trtexec --deploy=../googlenet_bvlc.prototxt  --model=../googlenet_bvlc.caffemodel  \
--output=prob --batch=1 --test=imagenet-validation/demofile.txt  --data_folder=imagenet-validation \
|& tee ./results_infer_trt_accuracy/result_googlenet_1_fp32.txt

LD_LIBRARY_PATH=./Model/opencv-3.4.0/build/lib/:$LD_LIBRARY_PATH ./bin/trtexec --deploy=../googlenet_bn_deploy.prototxt  --model=../googlenet_bn_stepsize_6400_iter_1200000.caffemodel \
--output=prob --batch=1 --test=imagenet-validation/demofile_original_index.txt  --data_folder=imagenet-validation \
|& tee ./results_infer_trt_accuracy/result_googlenet_bn_1_fp32.txt

LD_LIBRARY_PATH=./Model/opencv-3.4.0/build/lib/:$LD_LIBRARY_PATH ./bin/trtexec --deploy=../ResNet-50-deploy.prototxt  --model=../ResNet-50-model.caffemodel  \
--output=prob --batch=1 --test=imagenet-validation/demofile.txt  --data_folder=imagenet-validation \
|& tee ./results_infer_trt_accuracy/result_resnet50_1_fp32.txt

LD_LIBRARY_PATH=./Model/opencv-3.4.0/build/lib/:$LD_LIBRARY_PATH ./bin/trtexec --deploy=../ResNet-152-deploy.prototxt  --model=../ResNet-152-model.caffemodel  \
--output=prob --batch=1 --test=imagenet-validation/demofile.txt  --data_folder=imagenet-validation \
|& tee ./results_infer_trt_accuracy/result_resnet152_1_fp32.txt

LD_LIBRARY_PATH=./Model/opencv-3.4.0/build/lib/:$LD_LIBRARY_PATH ./bin/trtexec --deploy=../densenet121_deploy.prototxt  --model=../densenet121.caffemodel  \
--output=prob --batch=1 --test=imagenet-validation/demofile.txt  --data_folder=imagenet-validation  --scale \
|& tee ./results_infer_trt_accuracy/result_densenet121_1_fp32.txt
 
LD_LIBRARY_PATH=./Model/opencv-3.4.0/build/lib/:$LD_LIBRARY_PATH ./bin/trtexec --deploy=../squeezenet_v1.1_deploy.prototxt  --model=../squeezenet_v1.1.caffemodel  \
--output=prob --batch=1 --test=imagenet-validation/demofile.txt  --data_folder=imagenet-validation  \
|& tee ./results_infer_trt_accuracy/result_squeezenetv1_1_fp32.txt 
  


# fp16 
LD_LIBRARY_PATH=./Model/opencv-3.4.0/build/lib/:$LD_LIBRARY_PATH ./bin/trtexec --deploy=../googlenet_bvlc.prototxt  --model=../googlenet_bvlc.caffemodel  \
--output=prob --batch=1 --test=imagenet-validation/demofile.txt  --data_folder=imagenet-validation  --fp16 \
|& tee ./results_infer_trt_accuracy/result_googlenet_1_fp16.txt

LD_LIBRARY_PATH=./Model/opencv-3.4.0/build/lib/:$LD_LIBRARY_PATH ./bin/trtexec --deploy=../googlenet_bn_deploy.prototxt  --model=../googlenet_bn_stepsize_6400_iter_1200000.caffemodel \
--output=prob --batch=1 --test=imagenet-validation/demofile_original_index.txt  --data_folder=imagenet-validation --fp16\
|& tee ./results_infer_trt_accuracy/result_googlenet_bn_1_fp32.txt

LD_LIBRARY_PATH=./Model/opencv-3.4.0/build/lib/:$LD_LIBRARY_PATH ./bin/trtexec --deploy=../ResNet-50-deploy.prototxt  --model=../ResNet-50-model.caffemodel  \
--output=prob --batch=1 --test=imagenet-validation/demofile.txt  --data_folder=imagenet-validation --fp16 \
|& tee ./results_infer_trt_accuracy/result_resnet50_1_fp16.txt

LD_LIBRARY_PATH=./Model/opencv-3.4.0/build/lib/:$LD_LIBRARY_PATH ./bin/trtexec --deploy=../ResNet-152-deploy.prototxt  --model=../ResNet-152-model.caffemodel  \
--output=prob --batch=1 --test=imagenet-validation/demofile.txt  --data_folder=imagenet-validation --fp16 \
|& tee ./results_infer_trt_accuracy/result_resnet152_1_fp16.txt

LD_LIBRARY_PATH=./Model/opencv-3.4.0/build/lib/:$LD_LIBRARY_PATH ./bin/trtexec --deploy=../densenet121_deploy.prototxt  --model=../densenet121.caffemodel  \
--output=prob --batch=1 --test=imagenet-validation/demofile.txt  --data_folder=imagenet-validation  --scale --fp16 \
|& tee ./results_infer_trt_accuracy/result_densenet121_1_fp16.txt
 
LD_LIBRARY_PATH=./Model/opencv-3.4.0/build/lib/:$LD_LIBRARY_PATH ./bin/trtexec --deploy=../squeezenet_v1.1_deploy.prototxt  --model=../squeezenet_v1.1.caffemodel  \
--output=prob --batch=1 --test=imagenet-validation/demofile.txt  --data_folder=imagenet-validation  --fp16 \
|& tee ./results_infer_trt_accuracy/result_squeezenetv1_1_fp16.txt 