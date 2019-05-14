# CNN-Tensorflow  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | fp16|3518.0|4445.0|4548.0|  
| Resnet50 | fp16|2613.0|3256.0|3633.0|  
| Resnet152 | fp16|1356.0|1651.0|1854.0|  
| Densenet121 | fp16|1582.0|2099.0|2433.0|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
# CNN-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | fp16|6450.78|8676.37|9906.92|  
| Resnet50 | fp16|4418.10|5512.71|6309.89|  
| Resnet152 | fp16|1741.61|2096.42|2421.08|  
| Densenet121 | fp16|1708.15|2190.18|2531.49|  
| Squeezenetv1.1 | fp16|17829.50|22670.44|26633.60|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
# SSD-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| SSD-VGG16 | fp16|896.258|948.964|981.369|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
