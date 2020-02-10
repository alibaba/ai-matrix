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

if [ -d CNN_Tensorflow/results_train ]; then
	cp -r CNN_Tensorflow/results_train results/results_cnn_train
else
	echo "CNN_Tensorflow/results_train does not exist, check if test is ran successfully"
fi

if [ -d CNN_Tensorflow/results_infer ]; then
	cp -r CNN_Tensorflow/results_infer results/results_cnn_infer
else
	echo "CNN_Tensorflow/results_infer does not exist, check if test is ran successfully"
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

if [ -d DeepInterest/results ]; then
	cp -r DeepInterest/results results/results_deepinterest
else
	echo "DeepInterest/results does not exist, check if test is ran successfully"
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

if [ -d Mask_RCNN/results ]; then
	cp -r Mask_RCNN/results results/results_maskrcnn
else
	echo "Mask_RCNN/results does not exist, check if test is ran successfully"
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

if [ -d NMT/results ]; then
	cp -r NMT/results results/results_nmt
else
	echo "NMT/results does not exist, check if test is ran successfully"
fi

# ### SSD_VGG16_Caffe ###
# if [ -f SSD_VGG16_Caffe/results/results_train.csv ]; then
# 	cp SSD_VGG16_Caffe/results/results_train.csv results/results_ssd_vgg16_train.csv
# else
# 	echo "SSD_VGG16_Caffe/results/results_train.csv does not exist, check if test is ran successfully"
# fi

# if [ -f SSD_VGG16_Caffe/results/results_infer.csv ]; then
# 	cp SSD_VGG16_Caffe/results/results_infer.csv results/results_ssd_vgg16_infer.csv
# else
# 	echo "SSD_VGG16_Caffe/results/results_infer.csv does not exist, check if test is ran successfully"
# fi

# if [ -d SSD_VGG16_Caffe/results ]; then
# 	cp -r SSD_VGG16_Caffe/results results/results_ssd_vgg16
# else
# 	echo "SSD_VGG16_Caffe/results does not exist, check if test is ran successfully"
# fi

### SSD_ResNet34_PyTorch ###
if [ -f SSD_ResNet34_PyTorch/results/results_train.csv ]; then
	cp SSD_ResNet34_PyTorch/results/results_train.csv results/results_ssd_resnet34_train.csv
else
	echo "SSD_ResNet34_PyTorch/results/results_train.csv does not exist, check if test is ran successfully"
fi

# if [ -f SSD_ResNet34_PyTorch/results/results_infer.csv ]; then
# 	cp SSD_ResNet34_PyTorch/results/results_infer.csv results/results_ssd_resnet34_infer.csv
# else
# 	echo "SSD_ResNet34_PyTorch/results/results_infer.csv does not exist, check if test is ran successfully"
# fi

if [ -d SSD_ResNet34_PyTorch/results ]; then
	cp -r SSD_ResNet34_PyTorch/results results/results_ssd_resnet34
else
	echo "SSD_ResNet34_PyTorch/results does not exist, check if test is ran successfully"
fi

### SSD_ResNet18_Caffe ###
if [ -f SSD_ResNet18_Caffe/results/results_train.csv ]; then
	cp SSD_ResNet18_Caffe/results/results_train.csv results/results_ssd_resnet18_train.csv
else
	echo "SSD_ResNet18_Caffe/results/results_train.csv does not exist, check if test is ran successfully"
fi

if [ -f SSD_ResNet18_Caffe/results/results_infer.csv ]; then
	cp SSD_ResNet18_Caffe/results/results_infer.csv results/results_ssd_resnet18_infer.csv
else
	echo "SSD_ResNet18_Caffe/results/results_infer.csv does not exist, check if test is ran successfully"
fi

if [ -d SSD_ResNet18_Caffe/results ]; then
	cp -r SSD_ResNet18_Caffe/results results/results_ssd_resnet18
else
	echo "SSD_ResNet18_Caffe/results does not exist, check if test is ran successfully"
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

if [ -d DSSD/results ]; then
	cp -r DSSD/results results/results_dssd
else
	echo "DSSD/results does not exist, check if test is ran successfully"
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

if [ -d NCF/results ]; then
	cp -r NCF/results results/results_ncf
else
	echo "NCF/results does not exist, check if test is ran successfully"
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

if [ -d DIEN/results ]; then
	cp -r DIEN/results results/results_dien
else
	echo "DIEN/results does not exist, check if test is ran successfully"
fi

### BERT_Tensorflow ###
if [ -f BERT_Tensorflow/results/results_train.csv ]; then
	cp BERT_Tensorflow/results/results_train.csv results/results_bert_train.csv
else
	echo "BERT_Tensorflow/results/results_train.csv does not exist, check if test is ran successfully"
fi

if [ -f BERT_Tensorflow/results/results_infer.csv ]; then
	cp BERT_Tensorflow/results/results_infer.csv results/results_bert_infer.csv
else
	echo "BERT_Tensorflow/results/results_infer.csv does not exist, check if test is ran successfully"
fi

if [ -d BERT_Tensorflow/results ]; then
	cp -r BERT_Tensorflow/results results/results_bert
else
	echo "BERT_Tensorflow/results does not exist, check if test is ran successfully"
fi

### Faster_RCNN ###
if [ -f Faster_RCNN/results/results_train.csv ]; then
	cp Faster_RCNN/results/results_train.csv results/results_faster_rcnn_train.csv
else
	echo "Faster_RCNN/results/results_train.csv does not exist, check if test is ran successfully"
fi

if [ -f Faster_RCNN/results/results_infer.csv ]; then
	cp Faster_RCNN/results/results_infer.csv results/results_faster_rcnn_infer.csv
else
	echo "Faster_RCNN/results/results_infer.csv does not exist, check if test is ran successfully"
fi

if [ -d Faster_RCNN/results ]; then
	cp -r Faster_RCNN/results results/results_faster_rcnn
else
	echo "Faster_RCNN/results does not exist, check if test is ran successfully"
fi

### WideDeep ###
if [ -f WideDeep/results/results_train.csv ]; then
	cp WideDeep/results/results_train.csv results/results_wide_deep_train.csv
else
	echo "WideDeep/results/results_train.csv does not exist, check if test is ran successfully"
fi

if [ -f WideDeep/results/results_infer.csv ]; then
	cp WideDeep/results/results_infer.csv results/results_wide_deep_infer.csv
else
	echo "WideDeep/results/results_infer.csv does not exist, check if test is ran successfully"
fi

if [ -d WideDeep/results ]; then
	cp -r WideDeep/results results/results_wide_deep
else
	echo "WideDeep/results does not exist, check if test is ran successfully"
fi

### CPN ###
if [ -f CPN/results/results_train.csv ]; then
	cp CPN/results/results_train.csv results/results_cpn_train.csv
else
	echo "CPN/results/results_train.csv does not exist, check if test is ran successfully"
fi

if [ -f CPN/results/results_infer.csv ]; then
	cp CPN/results/results_infer.csv results/results_cpn_infer.csv
else
	echo "CPN/results/results_infer.csv does not exist, check if test is ran successfully"
fi

if [ -d CPN/results ]; then
	cp -r CPN/results results/results_cpn
else
	echo "CPN/results does not exist, check if test is ran successfully"
fi

### SegLink ###
if [ -f SegLink/results/results_train.csv ]; then
	cp SegLink/results/results_train.csv results/results_seglink_train.csv
else
	echo "SegLink/results/results_train.csv does not exist, check if test is ran successfully"
fi

if [ -f SegLink/results/results_infer.csv ]; then
	cp SegLink/results/results_infer.csv results/results_seglink_infer.csv
else
	echo "SegLink/results/results_infer.csv does not exist, check if test is ran successfully"
fi

if [ -d SegLink/results ]; then
	cp -r SegLink/results results/results_seglink
else
	echo "SegLink/results does not exist, check if test is ran successfully"
fi

### CRNN ###
if [ -f CRNN/results/results_train.csv ]; then
	cp CRNN/results/results_train.csv results/results_crnn_train.csv
else
	echo "CRNN/results/results_train.csv does not exist, check if test is ran successfully"
fi

if [ -f CRNN/results/results_infer.csv ]; then
	cp CRNN/results/results_infer.csv results/results_crnn_infer.csv
else
	echo "CRNN/results/results_infer.csv does not exist, check if test is ran successfully"
fi

if [ -d CRNN/results ]; then
	cp -r CRNN/results results/results_crnn
else
	echo "CRNN/results does not exist, check if test is ran successfully"
fi

python process_results.py
