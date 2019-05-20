# CNN-Tensorflow  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | fp32|875.0|945.0|953.0|  
| Resnet50 | fp32|450.0|454.0|468.0|  
| Resnet152 | fp32|178.0|181.0|188.0|  
| Densenet121 | fp32|362.0|386.0|401.0|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
# CNN-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | fp32|1065.91|1152.18|1171.74|  
| Resnet50 | fp32|507.91|506.40|521.56|  
| Resnet152 | fp32|185.81|189.14|194.79|  
| Densenet121 | fp32|292.51|300.35|313.02|  
| Squeezenetv1.1 | fp32|2935.77|3155.76|3250.43|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
# SSD-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| SSD-VGG16 | fp32|67.8812|68.1665|69.5623|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
