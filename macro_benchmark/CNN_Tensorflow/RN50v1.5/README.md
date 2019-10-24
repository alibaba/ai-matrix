This repo is from https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/RN50v1.5
Please refer to original repo for any further questions.  
  
We collected for benchmark the state of art GPU performance. Please run the scripts in **scripts** folder  
`bash ./scripts/RUN_FP16_1GPU.sh`  


| GPUs  | Throughput (real data with FP16 AMP) | Throughput (real data,FP16 AMP,XLA) |
| ------------- | ------------- |---------------|
| 1  | 772  | 1255 |
| 2  | 1494 | 2252 |
