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
