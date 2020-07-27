# Deep Interest Evolution Network for Click-Through Rate Prediction
https://arxiv.org/abs/1809.03672
## prepare data
### method 1
You can get the data from amazon website and process it using the script
```
sh prepare_data.sh
```
### method 2 (recommended)
Because getting and processing the data is time consuming，so we had processed it and upload it for you. You can unzip it to use directly.
```
tar -jxvf data.tar.gz
mv data/* .
tar -jxvf data1.tar.gz
mv data1/* .
tar -jxvf data2.tar.gz
mv data2/* .
```
When you see the files below, you can do the next work. 
- cat_voc.pkl 
- mid_voc.pkl 
- uid_voc.pkl 
- local_train_splitByUser 
- local_test_splitByUser 
- reviews-info
- item-info
## train model
```
python train.py train [model name] 
```
The model blelow had been supported: 
- DNN 
- PNN 
- Wide (Wide&Deep NN) 
- DIN  (https://arxiv.org/abs/1706.06978) 
- DIEN (https://arxiv.org/pdf/1809.03672.pdf) 

Note: we use tensorflow 1.4.

## Run the Benchmark
To run the benchmark, just run
```
./prepare_dataset.sh
./run.sh
```
prepare_dataset.sh only needs to be ran once.

To run training only, use
```
./train.sh
```
and to run inference only, use
```
./infer.sh
```

Currently AI Matrix only supports running on one accelerator device, being a GPU or other AI accelerator. If you intend to test an NVIDIA GPU, assign the GPU device before you start to run the benchmark suite using the following command:
```
export CUDA_VISIBLE_DEVICES="DEVICE_NUMBER_TO_TEST"
```
## Run the benchmark with synthetic input
The background is that real industry models usually have huge size of embedding table. They can not fit into GPU memory. Therefore, they are put into CPU ram and let CPU to handle the corresponding embedding lookup and gradients update. However, PCIE between CPU and GPU are the bottleneck in most cases. Industry solution needs to be innovative to well balance the issue here to achieve better performance. 
We initiate an synthetic random data as input to our model as a solution to mimic the real case where embedding table is huge. In this way, the embedding related operation can be chosen to implement on CPU side.
```
# embedding on GPU
python script/train.py --mode=synthetic --batch_size=32 --model=DIEN --embedding_device=gpu

# embedding on CPU
python script/train.py --mode=synthetic --batch_size=32 --model=DIEN --embedding_device=cpu
```
