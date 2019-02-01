Please follow a few steps below:  
1. download the tensorRT docker from Nvidia NGC. (The opencv compile may not work with higher CUDA like 10.0)  
docker pull nvcr.io/nvidia/tensorrt:18.08-py3

2. download the validation dataset and start the build of opencv and tensorrt demo 
git clone https://github.com/aimatrix-alibaba/imagenet-validation.git  
cd imagenet-validation  
bash uncompress.sh 
cd ..
build.sh 
  
3. run the program. Please config the parameter by yourself  
trtexec --deploy=model_deploy.prototxt  --model=model.caffemodel  --output=prob --batch=1 --test=imagenet-validation/demofile.txt --data_folder=imagenet-validation 

ex: 
googlenet_bvlc fp32  
LD_LIBRARY_PATH=./Model/opencv-3.4.0/build/lib/:$LD_LIBRARY_PATH ./bin/trtexec --deploy=../googlenet_bvlc.prototxt  --model=../googlenet_bvlc.caffemodel  --output=prob --batch=1 --test=imagenet-validation/demofile.txt  --data_folder=imagenet-validation/
  
googlenet_bvlc fp16  
LD_LIBRARY_PATH=./Model/opencv-3.4.0/build/lib/:$LD_LIBRARY_PATH ./bin/trtexec --deploy=../googlenet_bvlc.prototxt  --model=../googlenet_bvlc.caffemodel  --output=prob --batch=1 --test=imagenet-validation/demofile.txt  --data_folder=imagenet-validation/  --fp16
  
Please check ***infer-all-trt.sh*** for details.  
 

# Model accuracy test   
*numbers will differ from reference as different preprocessing method used*

|   fp32       | Validation Top1 | Validation Top5 | 
|--------------|-----------------|-----------------|
| Googlenet    |68.91            |89.15            |                    
| Googlenet-bn |72.00            |90.84            |                    
| Resnet50     |72.87            |91.14            |                       
| Resnet152    |74.93            |92.21            |                    
| Densenet121  |71.57            |90.39            |
| SqueezeNet   |58.36            |81.03            |

|   fp16       | Validation Top1 | Validation Top5 |                    
|--------------|-----------------|-----------------|
| Googlenet    |68.92            |89.14            |                    
| Googlenet-bn |72.01            |90.84            |                           
| Resnet50     |72.86            |91.15            |                
| Resnet152    |74.94            |92.22            |                
| Densenet121  |73.81            |91.83            |            
| SqueezeNet   |58.36            |81.02            |                    |

| Model          | scale | minus mean                     | crop                                   |
|----------------|-------|--------------------------------|----------------------------------------|
| Googlenet      | 1     | (B,G,R)=(103.94,116.78,123.68) | resize to 256x256 then crop to 224x224 |
| Googlenet-bn   | 1     | (B,G,R)=(103.94,116.78,123.68) | resize to 256x256 then crop to 224x224 |
| Resent50       | 1     | (B,G,R)=(103.94,116.78,123.68) | resize to 256x256 then crop to 224x224 |
| Resnet152      | 1     | (B,G,R)=(103.94,116.78,123.68) | resize to 256x256 then crop to 224x224 |
| Densenet121    | 0.017 | (B,G,R)=(103.94,116.78,123.68) | resize to 256x256 then crop to 224x224 |
| Squeezenetv1.1 | 1     | (B,G,R)=(103.94,116.78,123.68) | resize to 256x256 then crop to 224x224 |

**based on P100 + tensorRT 18.08 py3 docker image**  
