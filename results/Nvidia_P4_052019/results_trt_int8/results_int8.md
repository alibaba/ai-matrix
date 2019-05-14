#CNN-Tensorflow  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | int8|2602.0|3468.0|5074.0|  
| Resnet50 | int8|2550.0|3199.0|3517.0|  
| Resnet152 | int8|1163.0|1487.0|1664.0|  
| Densenet121 | int8|1571.0|1993.0|2269.0|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
#CNN-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | int8|6867.19|5194.43|4755.47|  
| Resnet50 | int8|2478.54|2236.25|2688.30|  
| Resnet152 | int8|1402.03|789.01|1609.09|  
| Densenet121 | int8|822.77|1102.24|1825.14|  
| Squeezenetv1.1 | int8|20262.24|26807.41|30980.88|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
#SSD-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| SSD-VGG16 | int8|213.038|215.882|217.174|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
