#!/bin/bash
CR_PWD=$(pwd)
TR_LOC=$CR_PWD/$1
DEVICE=1
DT=$2
MODEL="alexnet googlenet vgg16 resnet50"

mkdir results
cd results

for md in $MODEL; do
  echo "Inference on" $md  " .........."
  # warm up
  $TR_LOC/bin/giexec --deploy=$CR_PWD/${md}_deploy.prototxt  --output=prob --batch=16 --device=$DEVICE --model=$CR_PWD/${md}_iter_500.caffemodel 
  for bs in 16 32 64; do
      if [ "$DT" == "fp32" ]; then
          $TR_LOC/bin/giexec --deploy=$CR_PWD/${md}_deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/${md}_iter_500.caffemodel | tee ${md}_${bs}_fp32.txt
      elif [ "$DT" == "fp16" ]; then
          $TR_LOC/bin/giexec --deploy=$CR_PWD/${md}_deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/${md}_iter_500.caffemodel --half2 | tee ${md}_${bs}_fp16.txt
      elif [ "$DT" == "int8" ]; then
          $TR_LOC/bin/giexec --deploy=$CR_PWD/${md}_deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/${md}_iter_500.caffemodel --int8 | tee ${md}_${bs}_int8.txt
      else 
          exit 2
      fi
  done
done


MODEL="resnet152 densenet121 squeezenet_v1.1"
for md in $MODEL; do
  # warm up
  $TR_LOC/bin/giexec --deploy=$CR_PWD/${md}_deploy.prototxt  --output=prob --batch=16 --device=$DEVICE --model=$CR_PWD/${md}.caffemodel 
  echo "Inference on" $md  " .........."
  for bs in 16 32 64; do
      if [ "$DT" == "fp32" ]; then
          $TR_LOC/bin/giexec --deploy=$CR_PWD/${md}_deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/${md}.caffemodel | tee ${md}_${bs}_fp32.txt
      elif [ "$DT" == "fp16" ]; then
          $TR_LOC/bin/giexec --deploy=$CR_PWD/${md}_deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/${md}.caffemodel --half2 | tee ${md}_${bs}_fp16.txt
      elif [ "$DT" == "int8" ]; then
          $TR_LOC/bin/giexec --deploy=$CR_PWD/${md}_deploy.prototxt  --output=prob --batch=$bs --device=$DEVICE --model=$CR_PWD/${md}.caffemodel --int8 | tee ${md}_${bs}_int8.txt
      else 
          exit 2
      fi
  done
done
