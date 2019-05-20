# CNN-Tensorflow  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | int8|3276.0|3664.0|3769.0|  
| Resnet50 | int8|2478.0|2760.0|2795.0|  
| Resnet152 | int8|1324.0|1466.0|1452.0|  
| Densenet121 | int8|1546.0|1686.0|1745.0|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
# CNN-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | int8|6148.01|6815.69|6891.00|  
| Resnet50 | int8|4038.99|4425.34|4346.32|  
| Resnet152 | int8|1666.58|1820.06|1814.88|  
| Densenet121 | int8|1305.93|1385.30|1404.78|  
| Squeezenetv1.1 | int8|12558.18|14204.23|15125.20|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
# SSD-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| SSD-VGG16 | int8|465.36|457.437|455.529|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
