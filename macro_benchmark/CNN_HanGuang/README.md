# MLPerf Inference on HanGuangAI 
This test is referred from submission to MLPerf Inference v0.5. 

## Environment Setup

### Install OpenCV
   ```
   sudo apt install libopencv-dev   #For ubuntu
   sudo yum install opencv-devel    #For centos
   ```   
### Set up Python 3 Virtual Environment. Use python 3.5 in this test. Check [2] for CentOS 7
   ```
   python3 -m venv py3
   source py3/bin/activate
   ```
### Install glog
   ```
   sudo apt-get install libgflags-dev libgoogle-glog-dev   # For ubuntu
   sudo yum install gflags-devel glog-devel                # For centos 7
   ```
### Install libjpeg
   ```
   sudo apt-get install libjpeg-dev                        # For ubuntu
   sudo yum install libjpeg-turbo-devel                    # For centos 7
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
   pip install absl-py
   pip install protobuf
   ```

### Install Bazel
Bazel 0.20 or 0.19 is needed in this test. 
For ubuntu please check Bazel instructions to install.
For **CentOS 7**, we need to use bazel 0.19 or 0.20. It needs to build from scratch. Please follow instructions below. https://docs.bazel.build/versions/master/install-compile-source.html#bootstrap-bazel
You may need to install JDK 
   ```
   sudo yum install java-1.8.0-openjdk-devel
   ```
For convenience we recommend copying this binary to a directory that’s on your PATH (such as /usr/local/bin on Linux).
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
   cd ..           # out to root folder
   ```

## MLPerf Inference

### Clone the source code

   ``` 
   mkdir loadgen
   cd loadgen
   git clone --recursive git@github.com:mlperf/inference.git 
   cd ..
   ```
### Download the resnet model
  ```
  mkdir model
  cd model
  wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
  mv resnet50_v1.pb to fp32.pb
  cd ..
  ```
### Download the imagenet dataset

As indicated [here](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection#datasets) in MLPerf inference source code 
  ```
  mkdir dataset
  cd dataset
  # download dataset, you can either use above method or use a copy from yourself
  cd ..
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
    PWD=`pwd`
    export PY3_PATH=$PWD/py3
    export LG_PATH=$PWD/loadgen/inference/loadgen
    export TF_PATH=$PWD/tensorflow-1.13.1
    export MODEL_DIR=$PWD/model
    export DATA_DIR=$PWD/dataset
    export LD_LIBRARY_PATH=$PY3_PATH/lib/python3.5/site-packages/ratelnn/lib:$TF_PATH/bazel-bin/tensorflow
    ```

### Build the C++ Test Harness, GCC v5 up and cmake 3 is needed.Check [1,3] for CentOS 7
   
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





================
# Caveats for Centos7 installation
1. The gcc5 or up is needed. Please use scl project to install  
https://juejin.im/post/5d0ef5376fb9a07ef63fe74e  
2. Python3 is also needed in this test  
https://linuxize.com/post/how-to-install-python-3-on-centos-7/  
3. CMake 3 is needed
   ```
   sudo yum install epel-release
   sudo yum install cmake3
   ```
