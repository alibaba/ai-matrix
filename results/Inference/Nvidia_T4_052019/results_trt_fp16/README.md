# CNN-Tensorflow  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | fp16|2099.0|2345.0|2406.0|  
| Resnet50 | fp16|1540.0|1643.0|1723.0|  
| Resnet152 | fp16|753.0|811.0|863.0|  
| Densenet121 | fp16|953.0|1046.0|1073.0|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
# CNN-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | fp16|3242.03|3529.38|3669.37|  
| Resnet50 | fp16|2247.84|2407.75|2507.97|  
| Resnet152 | fp16|902.35|974.87|1023.20|  
| Densenet121 | fp16|873.03|919.52|936.47|  
| Squeezenetv1.1 | fp16|7498.96|8253.51|9108.58|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
# SSD-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| SSD-VGG16 | fp16|359.915|355.066|350.77|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
