## Usage  
1. Install git lfs by following the link: https://github.com/git-lfs/git-lfs/wiki/Installation
2. Compile the code.
   make
3. run setup.sh to download the weights data and prototxt model files.  
   ./setup.sh   
4. run infer.sh to run the infer tests with batch size=16,32,64 and precision=fp32,fp16,int8
   ./infer-fp32.sh  //fp32
   ./infer-fp16.sh //fp16
   ./infer-int8.sh //int8

