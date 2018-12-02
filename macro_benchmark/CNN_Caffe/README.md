# CNN Caffe

In this part performs the inference on CNN classic models.   
First, download TensorRT and place it under caffe_inference folder.   

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
 

If it is not intended to test on GPU, please use original **caffe test** (without TensorRT) command to run these models.


# Model sources
|              | Validation Top1 | Validation Top5 |
|--------------|-----------------|-----------------|
| Googlenet-bn |                 |                 |
| Googlenet    |                 |                 |
| Resnet50     |                 |                 |
| Resnet152    |                 |                 |
| Densenet121  |                 |                 |
| SqueezeNet   |                 |                 |


# Special Thanks to the authors and organizations below:
Googlenet_bn: https://github.com/lim0606/caffe-googlenet-bn  
Googlenet_bvlc: https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet  
Resnet:https://github.com/KaimingHe/deep-residual-networks  
Squeezenet: https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1  
Densenet121: https://github.com/shicai/DenseNet-Caffe  
