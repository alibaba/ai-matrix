#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# convert caffe model
CUDA_VISIBLE_DEVICES=0 python dump_caffemodel_weights.py \
  --caffe_root ../../../SSD_VGG16_Caffe \
  --prototxt_path ../../../SSD_VGG16_Caffe/models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced_deploy.prototxt \
  --caffemodel_path ../../../SSD_VGG16_Caffe/models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel \
  --caffe_weights_path ../../model/VGG_ILSVRC_16_layers_weights.pkl

# # convert to tensorflow checkpoint
CUDA_VISIBLE_DEVICES=0 python convert_caffemodel_to_ckpt.py \
  --model_scope ssd/vgg16 \
  --ckpt_path ../../model/VGG_ILSVRC_16_layers_ssd.ckpt \
  --caffe_weights_path ../../model/VGG_ILSVRC_16_layers_weights.pkl
