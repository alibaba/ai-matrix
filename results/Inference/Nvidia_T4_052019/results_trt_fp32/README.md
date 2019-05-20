# CNN-Tensorflow  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | fp32|1045.0|1105.0|1118.0|  
| Resnet50 | fp32|497.0|520.0|521.0|  
| Resnet152 | fp32|189.0|196.0|196.0|  
| Densenet121 | fp32|475.0|507.0|515.0|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
# CNN-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | fp32|1283.17|1328.77|1348.51|  
| Resnet50 | fp32|565.18|571.44|585.05|  
| Resnet152 | fp32|201.33|203.18|211.43|  
| Densenet121 | fp32|404.21|422.27|431.28|  
| Squeezenetv1.1 | fp32|3785.15|4004.63|4089.88|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
# SSD-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| SSD-VGG16 | fp32|84.6892|83.5088|83.1176|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
