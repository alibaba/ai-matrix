# CNN Caffe
## Performance test  
In this part performs the inference on CNN classic models.   
First, download TensorRT or Download tensorRT docker image from Nvidia NGC.

Second, follow the script below to run this test in nvidia tensorRT. 
Use the command below to run inference of all the models. ***TensorRT\_folder*** is where is folder name of downloaded tensorRT. ***data\_type*** can be fp32, fp16 or int8.   
Usually we need to run fp32 and fp16 data type.
``` 
cd macro_benchmark/caffe_inference
./run_test_tensorRT.sh TensorRT_folder data_type 
```
Then run the command below to get the scale score based on Nvidia card you'd like to compare. ***num_type***=3  will compare all 3 data types while =2 will compare fp32 and fp16.  

```
python cal_stat.py --num_type 3 --compare P4  
```
or
```
python cal_stat.py --num_type 2 --compare V100
```
Third, the performance number is recorded in aimatrix/result folder  
  
  
# Accuracy test
Please refer the TensorRT_demo for accuracy test. The test is based on trtexec.cpp 
and being added the preprocessing step in the code. Please refer the README inside
 that folder for accuracy number.


# Special Thanks to the authors and organizations below:
Googlenet_bn: https://github.com/lim0606/caffe-googlenet-bn  
Googlenet_bvlc: https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet  
Resnet:https://github.com/KaimingHe/deep-residual-networks  
Squeezenet: https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1  
Densenet121: https://github.com/shicai/DenseNet-Caffe  
