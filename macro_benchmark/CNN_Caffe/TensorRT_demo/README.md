Please follow a few steps below:  
1. download the tensorRT docker from Nvidia NGC. (The opencv compile may not work with higher CUDA like 10.0)  
docker pull nvcr.io/nvidia/tensorrt:18.08-py3

2. download the validation dataset and start the build of opencv and tensorrt demo   
git clone https://github.com/aimatrix-alibaba/imagenet-validation.git 。
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
  
Attention: densenet121 needs the scale flag in the command  
 
