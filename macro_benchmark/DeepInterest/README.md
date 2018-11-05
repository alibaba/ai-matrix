# DeepInterest
Deep Interest Network for Click-Through Rate Prediction

## Introduction
This is an implementation of the paper [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978) Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Han Zhu, Ying Fan, Na Mou, Xiao Ma, Yanghui Yan, Xingya Dai, Junqi Jin, Han Li, Kun Gai

Thanks Jinze Bai and Chang Zhou.

Bibtex:
```sh
@article{Zhou2017Deep,
  title={Deep Interest Network for Click-Through Rate Prediction},
  author={Zhou, Guorui and Song, Chengru and Zhu, Xiaoqiang and Ma, Xiao and Yan, Yanghui and Dai, Xingya and Zhu, Han and Jin, Junqi and Li, Han and Gai, Kun},
  year={2017},
}
```

## Requirements
* Python >= 3.6.1
* NumPy >= 1.12.1
* Pandas >= 0.20.1
* TensorFlow >= 1.4.0 (Probably earlier version should work too, though I didn't test it)
* GPU with memory >= 10G

Before running the benchmark, please install tensorflow on your system. Please use tensorflow version 1.10. If you intend to run these benchmarks on NVIDIA GPU, please install cuda 9. Before you start to run the benchmark, make sure that tensorflow and cuda are ready to use. You are responsible for setting these up properly in your environment.

You can use the following command to automatically install all required software packages:
```
./setup.sh
```

## Download dataset and preprocess
* Step 1: Download the amazon product dataset of electronics category, which has 498,196 products and 7,824,482 records, and extract it to `raw_data/` folder.
```sh
mkdir raw_data/;
cd utils;
bash 0_download_raw.sh;
```
* Step 2: Convert raw data to pandas dataframe, and remap categorical id.
```sh
python 1_convert_pd.py;
python 2_remap_id.py
```

All the above steps can be done using one script. Just run the following command:
```
./prepare_dataset.sh
```

## Training and Evaluation
This implementation not only contains the DIN method, but also provides all the competitors' method, including Wide&Deep, PNN, DeepFM. The training procedures of all method is as follows:
* Step 1: Choose a method and enter the folder.
```
cd din;
```
Alternatively, you could also run other competitors's methods directly by `cd deepFM` `cd pnn` `cd wide_deep`,
and follow the same instructions below.

* Step 2: Building the dataset adapted to current method.
```
python build_dataset.py
```
* Step 3: Start training and evaluating using default arguments in background mode. 
```
python train.py >log.txt 2>&1 &
```
* Step 4: Check training and evaluating progress.
```
tail -f log.txt
tensorboard --logdir=save_path
```

All the above steps can be done using one script. Just run the following command:
```
./run.sh
```
The run.sh script runs both training and inference. To run only training, use
```
./train.sh
```
and to run only inference, use
```
./infer.sh
```
Note that inference needs the checkpoints dumped by training as input, thus, to run inference training must be ran first.

## Results
Besides running training and inference, the run scripts also automatically generate the benchmark results. The results are dumped in results directory. results_train.csv and results_infer.csv summarizes the results of all batch sizes.

## Run the Benchmark - Summary
In summary, to run the benchmark, just run
```
./setup.sh
./prepare_dataset.sh
./run.sh
```
setup.sh and prepare_dataset.sh only need to be ran once.

Currently AI Matrix only supports running on one accelerator device, being a GPU or other AI accelerator. If you intend to test an NVIDIA GPU, assign the GPU device before you start to run the benchmark suite using the following command:
```
export CUDA_VISIBLE_DEVICES="DEVICE_NUMBER_TO_TEST"
```

## Dice
There is also an implementation of Dice in folder 'din', you can try dice following the code annotation in `din/model.py` or replacing model.py with model\_dice.py
