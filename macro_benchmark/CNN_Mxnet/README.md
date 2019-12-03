Resnet v1.5 Mxnet  
The repo contains distributed training code of Resnet50 v1.5 reference to [NV submission to MLPerf](https://github.com/mlperf/training_results_v0.6/tree/master/NVIDIA/benchmarks/resnet/implementations/mxnet).
  
# Requirements
nvidia-docker or docker 19.03 with cuda support
MXNet 18.11-py3 NGC container or new version 
Imagenet dataset

# Directions 
## 1. Download imagenet 
Please download the dataset manually following the instructions from the ImageNet website. We use non-resized Imagenet dataset, packed into MXNet recordio database. It is **not** resized and **not** normalized. No preprocessing was performed on the raw ImageNet jpegs.

For further instructions, see https://github.com/NVIDIA/DeepLearningExamples/blob/master/MxNet/Classification/RN50v1.5/README.md#prepare-dataset 

## 2. Set up docker image
docker build --network=host --pull -t mlperf-nvidia:image_classification .
Or use we build
docker pull reg.docker.alibaba-inc.com/ai_matrix/mlperf-nvidia:image_classification_1910  

## 3. Set up docker container on each node with passwordless authentication
1) set up docker container
 sudo docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v /work1/:/data --network=host --ipc=host --uts=host --security-opt seccomp=unconfined --name=image1910 reg.docker.alibaba-inc.com/ai_matrix/mlperf-nvidia:image_classification_1910
  
2) Inside container:
apt-get update
apt-get install openssh-server
mkdir -p /var/run/sshd
ssh-keygen -t rsa

3) Allow root ssh access and add port number 12345 as the specific port we will use 
sed -i 's+#PermitRootLogin prohibit-password+PermitRootLogin yes+g' /etc/ssh/sshd_config
sed -i 's+#Port 22+Port 22 \nPort 12345+g' /etc/ssh/sshd_config
service ssh restart  #restart sshd service

4) Add the **host** node's ~/.ssh/id_rsa.pub to each **worker** node's ~/.ssh/authorized_keys


  
## 4. Run the training process with different configurations
We tested 1/2/4 nodes (each node has 4 V100-PCIE cards) in our lab environment. The convergence accuracy is set to 75.9%    
  
Nodes | Converge epochs| Ending epochs | Time to converge (min)
------------- | ------------- | ------------ | -------------
1 node        | 40            | 44            | 163.77        
2 nodes       | 40            | 44            | 84.10         
4 nodes       | 40            | 44            | 42.68         


The scripts to run different config are at /scripts folder. The ip address in the scripts need to be modified to fit your environment.
