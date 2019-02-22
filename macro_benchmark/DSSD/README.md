# DSSD : Deconvolutional Single Shot Detector

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

By [Cheng-Yang Fu*](http://www.cs.unc.edu/~cyfu/), [Wei Liu*](http://www.cs.unc.edu/~wliu/), Ananth Ranga, Ambrish Tyagi, [Alexander C. Berg](http://acberg.com).

*=Equal Contribution

### Status now 
The first version is done. Users can start training the SSD/DSSD with Resnet-101 now. 

Stay tuned. Models, trained for Pascal VOC 2007, 2012 and COCO , will be released soon. 

### Introduction

Deconvolutional SSD brings additional context into state-of-the-art general object detection by adding extra deconvolution structures. The DSSD achieves much better accuracy on small objects compared to SSD.

The code is based on [SSD](https://github.com/weiliu89/caffe/tree/ssd). For more details, please refer to our [arXiv paper](https://arxiv.org/abs/1701.06659). 

### Citing DSSD

Please cite DSSD in your publications if it helps your research:

    @inproceedings{Fu2016dssd,
      title = {{DSSD}: Deconvolutional Single Shot Detector},
      author = {Fu, Cheng-Yang and Liu, Wei and Ranga, Ananth and Tyagi, Ambrish and Berg, Alexander C.},
      booktitle = {arXiv preprint arXiv:1701.06659},
    }


### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [COCO_Models](#cocomodels)
5. [Run the benchmark](#runbenchmark)

### Installation
1. Download the code from github. We call this directory as `$CAFFE_ROOT` later.

	```Shell
	git clone https://github.com/chengyangfu/caffe.git
	cd $CAFFE_ROOT
	git checkout dssd
	```
	
2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.

	```Shell
  	# Modify Makefile.config according to your Caffe installation.
  	cp Makefile.config.example Makefile.config
  	make -j8
  	# Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  	make py
  	make test -j8
  	# (Optional)
  	make runtest -j
  	```

### Preparation
1.  Please Follow the Orginal [SSD](https://github.com/weiliu89/caffe/tree/ssd) to do all the preparation works. You should have lmdb fils for VOC2007. Check the following two links exist or not. 
   
   	```Shell
   	ls $CAFFE_ROOT/examples
   	# $CAFFE_ROOT/examples/VOC0712/VOC0712_trainval_lmdb
   	# $CAFFE_ROOT/examples/VOC0712/VOC0712_test_lmdb
   	```
   
2.  Download the Resnet-101 models from the [Deep-Residual-Network](https://github.com/KaimingHe/deep-residual-networks).
    
	```Shell
	# creat the directory for ResNet-101
	cd $CAFFE_ROOT/models
	mkdir ResNet-101
	# Rename the Resnet-101 models and put in the ResNet-101 direcotry
	ls $CAFFE_ROOT/models/ResNet-101
	# $CAFFE_ROOT/models/ResNet-101/ResNet-101-model.caffemodel
	# $CAFFE_ROOT/models/ResNet-101/ResNet-101-deploy.prototxt
	```

### Train/Eval
1. Train and Eval the SSD model 

	```Shell
	# Train the SSD-ResNet-101 321x321
	python examples/ssd/ssd_pascal_resnet_321.py
	# GPU setting may need be change according to the numbers of gpu 
	# models are generated in:
	# $CAFFE_ROOT/models/ResNet-101/VOC0712/SSD_VOC07_321x321
	# Evaluate the model
	cd $CAFFE_ROOT
	./build/tools/caffe train --solver="./models/ResNet-101/VOC0712/SSD_VOC07_321x321/test_solver.prototxt"  \
	--weights="./models/ResNet-101/VOC0712/SSD_VOC07_321x321/ResNet-101_VOC0712_SSD_VOC07_321x321_iter_80000.caffemodel" \
	--gpu=0
	# batch size in the test.prototxt may need be changed.
	# If the batch size is changed, remeber to change the test_iter in test_solver.prototxt at same time. 
	# It should reach 77.5* mAP at 80k iterations.
	```
   
2. Train and Evaluate the DSSD model. In this script, Resnet-101 and SSD related layers are frozen and only the DSSD related layers are trained.

	```Shell
	# Use the SSD-ResNet-101 321x321 as the pretrained model
	python examples/ssd/ssd_pascal_resnet_deconv_321.py
	# Evaluate the model
	cd $CAFFE_ROOT
	./build/tools/caffe train --solver="./models/ResNet-101/VOC0712/DSSD_VOC07_321x321/test_solver.prototxt"  \
	--weights="./models/ResNet-101/VOC0712/DSSD_VOC07_321x321/ResNet-101_VOC0712_DSSD_VOC07_321x321_iter_30000.caffemodel" \
	--gpu=0
	# It should reach 78.6* mAP at 30k iterations.
	```
	
3. Train and Evalthe DSSD model. In this script, we try to fine-tune the entire network. In order to sucessfully finetune the network, we need to freeze all the batch norm related layers in Caffe.

	```Shell
	# Use the DSSD-ResNet-101 321x321 as the pretrained model
	python examples/ssd/ssd_pascal_resnet_deconv_ft_321.py
	# Evaluate the model
	cd $CAFFE_ROOT
	./build/tools/caffe train --solver="./models/ResNet-101/VOC0712/DSSD_VOC07_FT_321x321/test_solver.prototxt"  \
	--weights="./models/ResNet-101/VOC0712/DSSD_VOC07_FT_321x321/ResNet-101_VOC0712_DSSD_VOC07_FT_321x321_iter_40000.caffemodel" \
	--gpu=0
	# Finetuning the entire network only works for the model with 513x513 inputs not 321x321. 
	```
  
### COCO_Models
1. We add two scripts for training SSD/DSSD with 513x513 inputs on COCO. 
  	
	```Shell
	# Train SSD513-ResNet101 on COCO 
	python examples/ssd/ssd_coco_resnet_513.py
	# Train DSSD513-ResNet101 on COCO and use SSD513 as the pretrained model
	python examples/ssd/ssd_coco_resnet_deconv_513.py
	```
2. We strongly suggest to use the trained models instead of training from scracth. 
		
	[SSD_513_COCO](https://drive.google.com/file/d/0By9LEMeCDdboa0IxSkIxbEVWZVk/view?usp=sharing)
	
	[DSSD_513_COCO](https://drive.google.com/file/d/0By9LEMeCDdboSDRlVHY2SFNJVzQ/view?usp=sharing) 
	```Shell
	# move the compressed files at $CAFFE_ROOT/models/ResNet-101
	cd $CAFFE_ROOT/models/ResNet-101
	tar -vzxf SSD_513_COCO.tar.gz
	tar -vzxf DSSD_513_COCO.tar.gz
	```
	P.S.: Please change the field "start" to offset" in PriorBox Layers.
  
3. In our experiments, the model with 513x513 inputs are trained using NVIDIA P40 which consists of 22GB memory. Because we add extra batch normalization layers, it's important to make the mini-batchs size at least 5 in each gpu. So, if you use the gpu with smaller memory, I don't think you can replicate the results.

### Run the benchmark
Automation scripts are provided to easily run the benchmark in just a few scripts. There are three sripts for running the benchmark: setup.sh, prepare_dataset.sh, and run.sh. setup.sh compiles caffe and the benchmark. prepare_dataset.sh downloads the VOC2007 and VOC2012 dataset needed in both training and inference and processes the dataset if necessary. run.sh runs both training and inference in one sript and automatically generates results.

To run the benchmark, use the following commands:
```
./setup.sh
./prepare_dataset.sh
./run.sh
```
setup.sh and prepare_dataset.sh only need to be ran once, and once the benchmark and the dataset are set up there is no need to run them a second time.

The run.sh script runs both training and inference together. If you want to run training or inference seperately, use
```
./train.sh
```
to run training and use
```
./infer.sh
```
to run inference.
