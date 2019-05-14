#CNN-Tensorflow  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | fp16|2290.0|2730.0|2907.0|  
| Resnet50 | fp16|1429.0|1580.0|1634.0|  
| Resnet152 | fp16|629.0|672.0|700.0|  
| Densenet121 | fp16|1081.0|1246.0|1312.0|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
#CNN-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | fp16|3425.90|3994.53|4319.22|  
| Resnet50 | fp16|1842.41|1957.94|2045.45|  
| Resnet152 | fp16|698.52|736.19|765.41|  
| Densenet121 | fp16|909.63|1006.07|1058.17|  
| Squeezenetv1.1 | fp16|10041.99|11326.67|12154.18|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
#SSD-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| SSD-VGG16 | fp16|285.781|294.026|298.143|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
