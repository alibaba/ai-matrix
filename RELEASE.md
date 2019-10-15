# Release 1.0.5

## Major Features and Improvements  
* Updated the Tensorflow docker to 19.09
* Added SSD with ResNet34 as backbone in PyTorch framework
* Removed SSD with VGG16 in Caffe framework
* Added support of running benchmarks in detached docker mode
* Moved large files to Zenodo and reduced repository size

# Release 1.0.4  

## Major Features and Improvements  
* Add TF inference test with integration of TensorRT  
* Add DIEN model from Alimama  
* Add BERT model, a GOOGLE version and an NVIDIA version
* Add cascaded pyramid network
* Add convolutional recurrent neural network
* Add SSD with resnet18 as backbone uisng Caffe framework
* Add SSD with VGG18 as backbone using Caffe framework
* Add TensorRT implementation of SSD with VGG18 as backbone
* Add Faster RCNN
* Add Graph Convolutional Network
* Add NMT TensorRT implementation
* Add SegLink model
* Add Wide & Deep model
* Change unit of results to measurable metrics such as images/s, recommendations/s, sentences/s
* Fix many miscellaneous issues
* Refine scripts for accuracy tests

# Release 1.0.3

## Major Features and Improvements
* Add new models: NCF for recommendation class, DSSD for object detection class
* Nvidia docker has license issue on distribution, users have to download by themselves. Add script to install some dependencies
* Add md5 checksum for some big files to help us spot the download issues
* Add multi-card training for DIN model
* Add accuracy test cases in Caffe CNN models. Inference engine from different vendor could compare not only performance number but also accuracy loss


# Release 1.0.2 

## Major Features and Improvements  
* Add the trained checkpoint file for googlenet, resnet50, resnet152, densenet121  
* Add multi-card training in for CNN-Tensorflow, SSD, MaskRCNN, NMT  


# Release 1.0.1

## Major Features and Improvements
* Reorganize the automation workflow to improve the running scripts quality.
* Users can choose run all of application in a few scripts or each application separately.
* Add preprocessing script to extract and save data to csv file.
* Remove Alexnet as it is out of date.
* Remove Vgg16 as it is repeatedly used in SSD test.
* Add TensorRT-5 inference script for Caffe model.
* Add Tensorcore FP16 GEMM in micro tests.

## Bug Fixes and Other Changes
DIN model:
* Change the inference workload to apply 100 items for each user to recommend. 
* The inference batch size is based on number of users. It is set to 1, 32, and 64. Iteration of 1000 is applied to minimize the overhead.
