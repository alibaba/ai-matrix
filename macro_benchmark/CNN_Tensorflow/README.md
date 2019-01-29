# CNN Tensorflow

The CNN test contains the following image recognition models: densenet121, googlenet, resnet152, resnet50, synNet. synNet is a synthetic CNN model generated using AI Matrix's synthetic framework.

## System Requirements
Before running the benchmarks, please install tensorflow on your system. Please use tensorflow version 1.10. If you intend to run these benchmarks on NVIDIA GPU, please install cuda 9. Before you start to run the benchmarks, make sure that tensorflow and cuda are ready to use. You are responsible for setting these up properly in your environment.

## Run the Benchmark
Automation scripts are provided to easily run all the models in just a few scripts. There are two sripts for running the benchmark: setup.sh, and run.sh.

setup.sh installs all the software packages required by the benchmarks.
run.sh runs both training and inference in one sript and automatically generates results.

Currently AI Matrix only supports running on one accelerator device, being a GPU or other AI accelerator. If you intend to test an NVIDIA GPU, assign the GPU device before you start to run the benchmark using the following command:
```
export CUDA_VISIBLE_DEVICES="DEVICE_NUMBER_TO_TEST"
```

To run the benchmark, use the following commands:
```
./setup.sh
./run.sh
```

setup.sh only needs to be ran once, and once the required packages are installed there is no need to run it a second time. run.sh can be run multiple times.

The run.sh script runs both training and inference together. If you want to run training or inference seperately, use
```
./train-all.sh
```
to run training and use
```
./infer-all.sh
```
to run inference. Note that inference needs the checkpoints dumped by training as input, thus, to run inference training must be ran first.

To run a single model for training, use
```
./train.sh MODEL_NAME
```
To run a single model for inference, use
```
./infer.sh MODEL_NAME
```
MODEL_NAME is any of densenet121, googlenet, resnet152, resnet50, synNet.

## Run Training on Multiple Accelerators
To run training on multiple accelerators, please set NUM_ACCELERATORS environment variable in your terminal before running the run.sh or train-all.sh scripts, e.g.,
```
export NUM_ACCELERATORS=8
```

## Results  
Besides running training and inference, the run scripts also automatically generate the benchmark results. The results are dumped in results_train and results_infer directories. results.csv in each directory tables the results of all models.

## Inference with TensorRT (real data), checkpoint model  
python nvcnn.py --model=MODEL  --batch_size=SIZE  --num_gpus=1  --display_every=100  --log_dir=/PATH/TO/CHECKPOINT  --eval  --data_dir=/PATH/TO/TFRECORD  --num_epochs=1 --use_trt --trt_precision=PRECISION 
  
## Inference with TensorRT (synthetic data), checkpoint model 
python nvcnn.py --model=MODEL  --batch_size=SIZE  --num_gpus=1  --display_every=100  --log_dir=/PATH/TO/CHECKPOINT  --eval  --num_epochs=1 --use_trt --trt_precision=PRECISION

## Inference with TensorRT (synthetic data), frozen model  
python nvcnn.py --model=MODEL  --batch_size=SIZE  --num_gpus=1  --display_every=100  --log_dir=/PATH/TO/CHECKPOINT  --eval  --num_epochs=1 --use_trt --trt_precision=PRECISION --cache --cache_path=/PATH/TO/FROZEN_MODEL
  
## Prerequisite of inference  
Please download the imagenet validation data and inference model weights first. The download script is located in ai-matrix/macro_benchmark/pretrained_models  
