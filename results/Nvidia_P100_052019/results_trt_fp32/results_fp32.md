#CNN-Tensorflow  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | fp32|1499.0|1760.0|1871.0|  
| Resnet50 | fp32|834.0|916.0|952.0|  
| Resnet152 | fp32|341.0|364.0|379.0|  
| Densenet121 | fp32|728.0|841.0|887.0|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
#CNN-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | fp32|1962.14|2254.08|2413.54|  
| Resnet50 | fp32|968.44|1042.95|1083.63|  
| Resnet152 | fp32|362.15|383.73|397.62|  
| Densenet121 | fp32|635.13|710.77|747.33|  
| Squeezenetv1.1 | fp32|6245.22|7124.17|7740.56|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
#SSD-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| SSD-VGG16 | fp32|166.776|170.596|170.828|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
