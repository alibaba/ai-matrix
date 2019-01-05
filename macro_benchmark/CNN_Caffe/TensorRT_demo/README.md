Please follow a few steps below:
1. download the validation dataset  
git clone git@github.com:aimatrix-alibaba/imagenet-validation.git  
cd imagenet-validation  
bash uncompress.sh  
 
2. compile the trt program
make
  
3. run the program. Please config the parameter by yourself  
trtexec --deploy=model_deploy.prototxt  --model=model.caffemodel  --output=prob --batch=1 --test=imagenet-validation/demofile.txt --data_folder=imagenet-validation 

Attention: densenet121 needs the scale flag in the command  
 
