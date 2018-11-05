# Macro Benchmarks

The macro benchmark category consists of some widely used deep learning neural networks in a number of common application fields. One or more representative applications are selected for each application field. The macro benchmark suite currently has the following tests: CNN, DeepInterest, Mask_RCNN, NMT, and SSD. The CNN test contains the following image recognition models: alexnet, densenet121, googlenet, resnet152, resnet50, synNet, and vgg16. synNet is a synthetic CNN model generated using AI Matrix's synthetic framework.

## System Requirements
Before running the benchmarks, please install tensorflow on your system. Please use tensorflow version 1.10. If you intend to run these benchmarks on NVIDIA GPU, please install cuda 9. Before you start to run the benchmarks, make sure that tensorflow and cuda are ready to use. You are responsible for setting these up properly in your environment.

## Run the Benchmark Suite
Automation scripts are provided to easily run all the tests in just a few scripts. In the macro_benchmark directory, there are three sripts for running the entire benchmark suite: setup.sh, prepare_dataset.sh, and run.sh.

setup.sh installs all the software packages required by the benchmarks.
prepare_dataset.sh downloads all the dataset needed in both training and inference and processes the dataset if necessary.
run.sh runs both training and inference in one sript and automatically generates results in excel sheets.

Currently AI Matrix only supports running on one accelerator device, being a GPU or other AI accelerator. If you intend to test an NVIDIA GPU, assign the GPU device before you start to run the benchmark suite using the following command:
```
export CUDA_VISIBLE_DEVICES="DEVICE_NUMBER_TO_TEST"
```

To run the entire benchmark suite, use the following commands:
```
cd macro_benchmark
./setup.sh
./prepare_dataset.sh
./run.sh
```

setup.sh and prepare_dataset.sh only need to be ran once, and once the required packages and dataset are set up there is no need to run these two scripts a second time. run.sh can be run multiple times. It takes about half an hour to download and process the dataset and three hours or longer to run the entire benchmark suite.

The run.sh script runs both training and inference together. If you want to run training or inference seperately, use
```
./run-train.sh
```
to run training and use
```
./run-infer.sh
```
to run inference. Note that for CNN_Tensorflow and DeepInterest, inference needs the checkpoints dumped by training as input, thus, to run the inference of these two tests training must be ran first. The run-infer.sh script will automatically run training first before inference if it detects that training has not been run for CNN_Tensorflow and DeepInterest.

## Run a Single Benchmark
The instructions to run a single benchmark can be found in the README.md file in each test directory.

## Results
Besides running training and inference, the run scripts also automatically generate the benchmark results. The results are dumped in macro_benchmark/results. There are two csv files for each test, one for training and one for inference. All results are summarized in two excel files, results_train.xlsx for training results and results_infer.xlsx for inference results.

A script is also provided to automatically compare the results of two systems. For example, if you have run the benchmark suite on one system with an NVIDIA P100 GPU and another system with an NVIDIA V100 GPU and the results are put in two directories P100 and V100 respectively, you can compare P100 performance with V100 performance by running the following command and get the normalized P100 performance over V100.
```
python compare_results.py --target_dir P100 --ref_dir V100
```
The normalized results are put in a folder named P100_vs_V100. See the compare_results.py script for detailed usage information.

## Other Tests
Except for the aforementioned five tests, there are also two other tests in the macro benchmark category: CNN_Caffe and DeepSpeech. The reason that these two tests are not put together with the aforementioned five is that they can only be ran on GPU accelerator, while the objective of AI Matrix is to support all kinds of AI accelerators, not only GPUs. Users can still run these two tests on systems with GPUs by following the guide in the README.md files in CNN_Caffe and DeepSpeech directories.

In the future, the support of all accelerators other than GPU will be added in these two tests.
