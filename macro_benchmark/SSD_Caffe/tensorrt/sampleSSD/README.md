## Usage  
1. Compile the code.
   make
2. run setup.sh to download the weights data and prototxt model files.  
   ./setup.sh   
3. run infer.sh to run the infer tests with batch size=16,32,64 and precision=fp32,fp16,int8
   ./infer-fp32.sh  //fp32
   ./infer-fp16.sh //fp16
   ./infer-int8.sh //int8

