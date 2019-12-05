# MLPerf Inference on HanGuangAI 
This test is referred from submission to MLPerf Inference v0.5. 

## Environment Setup

### Install OpenCV
   ```
   sudo apt install libopencv-dev
   ```   
### Set up Python 3 Virtual Environment
   ```
   python3 -m venv py3
   source py3/bin/activate
   ```
   
### Install Python Requisites
   ```
   pip install --upgrade pip
   pip install --upgrade setuptools
   pip install scipy
   pip install six numpy wheel setuptools mock 'future>=0.17.1'
   pip install keras_applications==1.0.6 --no-deps
   pip install keras_preprocessing==1.0.5 --no-deps
   pip install pudb  
   ```

### Verfiy Bazel version is 0.19.0 (which works with tensorflow 1.13.1)

   ``` 
   bazel version
   ```

### Download Tensorflow 1.13.1 Source Code and build it
   ``` 
   wget https://github.com/tensorflow/tensorflow/archive/v1.13.1.tar.gz
   tar xzvf v1.13.1.tar.gz
   cd tensorflow-1.13.1
   ./configure
   bazel build -c opt --copt=-msse4.2 --copt=-mavx2 --copt=-mfma --copt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow/tools/pip_package:build_pip_package
   bazel-bin/tensorflow/tools/pip_package/build_pip_package ../
   pip install ../tensorflow-1.13.1-cp35-cp35m-linux_x86_64.whl
   ```

### Build Tensorflow CC

   ``` 
   bazel build -c opt --copt=-msse4.2 --copt=-mavx2 --copt=-mfma --copt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow:libtensorflow_cc.so
   ```

## MLPerf Inference

### Clone the source code

   ``` 
   mkdir loadgen
   cd loadgen
   git clone --recursive git@github.com:mlperf/inference.git 
   ```
### Download the resnet model
  ```
  mkdir model
  cd model
  wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
  mv resnet50_v1.pb to fp32.pb
  ```
### Download the imagenet dataset

As indicated [here](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection#datasets) in MLPerf inference source code 
  ```
  mkdir dataset
  cd dataset
  ```
### HanGuangAI

* HanGuangAI is a software package developed by T-Head (PingTouGe) Semiconductor Co., Ltd., subsidiary of Alibaba Group Holding Ltd., connecting popluar frameworks to HanGuang NPU. It will be distributed to customer once HanGuang NPU is made public accessible. 
* Install HanGuang AI 1.0.3
  
    ``` 
    pip install hgai-ubuntu-1.0.3-cp35-cp35m-linux_x86_64.whl
    ```

### Environment Variables
* Add tensorflow cc library location to LD_LIBRARY_PATH
* Add path environment variables used by c++ test harness build

    ``` 
    export PY3_PATH=./py3
    export LG_PATH=./loadgen
    export TF_PATH=./tensorflow-1.13.1
    export MODEL_DIR=./model
    export DATA_DIR=./dataset
    export LD_LIBRARY_PATH=$PY3_PATH/lib/python3.5/site-packages/ratelnn/lib:$TF_PATH/bazel-bin/tensorflow
    ```

### Build the C++ Test Harness
   
    cd code/resnet/tfcpp/classification/cpp
    mkdir build
    cd build
    cmake ..
    make
   

## Quantization
   ```
   cd code/resnet/tfcpp/quantize
   python converter.py --output_type npu
   ```  
    
## Execution
   
### Performance Mode

   ``` 
   cd code/resnet/tfcpp/classification/cpp
   ./run_perf.sh Offline
   ```

Feel free to change the Scenarios with SingleStream, MultiStream, Server
The results locate in mlperf_log_summary.log
