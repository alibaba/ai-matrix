#CNN-Tensorflow  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | fp32|2007.0|2463.0|2738.0|  
| Resnet50 | fp32|1182.0|1333.0|1446.0|  
| Resnet152 | fp32|481.0|553.0|563.0|  
| Densenet121 | fp32|991.0|1195.0|1334.0|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
#CNN-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| Googlenet | fp32|2818.46|3406.90|3751.53|  
| Resnet50 | fp32|1431.79|1579.63|1692.71|  
| Resnet152 | fp32|520.84|594.53|608.65|  
| Densenet121 | fp32|967.23|1112.88|1230.96|  
| Squeezenetv1.1 | fp32|9143.85|10528.05|11742.30|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
#SSD-Caffe  
The test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  
  
| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  
|-----------|-----------|---------|---------|---------|  
| SSD-VGG16 | fp32|249.805|252.348|252.258|  
  
*BS: Batch Size*  
*Unit: Img/sec*  
  
