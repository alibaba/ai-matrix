# AIMatrix

This benchmarks focus on inference mainly at current stage. It consists of four parts with seperate focus on different workload characteristics. It consist of some widely used models in both academia and industry. As a Alibaba benchmark, we have included some innovative or widely deployed Alibaba AI algorithms in our benchmarks. The detailed statement of design of this benchmarks is [**here**](http://aimatrix.ai/#!/docs/goals.md?lang=en-us).  
The README is focused on usage of this benchmark suites.
 
It includes the CNN and RNN models. The purpose of this benchmarks to test the performance of complete models with focus on inference. The models weights are obtained by trained on synthetic data with a few hundreds iterations with initial random numbers (it is enough for performance testing purpose). They will be tested in two frameworks: caffe and tensorflow. Caffe models and weights are input to TensorRT and the results are based on TensorRT optimized graph. Tensorflow inference is tested in Tensorflow framework with training first to generate checkpoint files and these files can be used in inference test later.  
Our model database tracks the academia and industry model development all the time. New models could be added if they are satisfied with our selection standards.  The CNN models included in our benchmarks are  
1) googlenet  
2) vgg16  
3) resnet50  
4) resnet152  
5) densenet  

6) resnet101 reflected in Mask RCNN application  
  
RNN models are reflected in application below:  
1) natural machine translation (NMT) benchmark  
2) deepspeech  
  
## 1. Caffe inference
In this part performs the inference on CNN classic models.   
First, download TensorRT and place it under caffe_inference folder.   

Second, follow the script below to run this test in nvidia tensorRT. 
Use the command below to run inference of all the models. ***TensorRT\_folder*** is where is folder name of downloaded tensorRT. ***data\_type*** can be fp32, fp16 or int8.   
Usually we need to run fp32 and fp16 data type.
``` 
cd macro_benchmark/caffe_inference
./run_test_tensorRT.sh TensorRT_folder data_type 
```
Then run the command below to get the scale score based on Nvidia card you'd like to compare. ***num_type***=3  will compare all 3 data types while =2 will compare fp32 and fp16.  

```
python cal_stat.py --num_type 3 --compare P4  
```
or
```
python cal_stat.py --num_type 2 --compare V100
```
 

If it is not intended to test on GPU, please use original **caffe test** (without TensorRT) command to run these models. 
 
## 2. Tensorflow inference
In this part, similar CNN models can be tested under tensorflow framework. To run the inference tests, you need to run a few hundreds iterations of training tests to generate some checkpoint files. Then you could use our script to run the inference tests.  Please replace the models with _models_ available. It will run batch size = 16, 32 ,64. Feel free to change the batch size as you need.

To run fp32 inference:
```
cd macro_benchmark/tf_inference
./run-train-all.sh models
./run_infer-all.sh models
```
Similar to Caffe tests, you can compare the perf data with ***card***=P4 or V100 on fp32 data type:
```
python cal_stat.py --compare card
```
 
## 3. Mask RCNN inference
Here, we provide the Mask RCNNwith inference performance tests.
resnet101 is used as backbone in this test framework. 28 images are tested in the inference situation. Each image is duplicated to make batch size = n. So total batch size of image detected in this test is 28*n

Step 1: 
You need to install a few packages before running the test:
```
pip3 install scikit-image
pip3 install 'keras==2.1.6' 
pip3 install imgaug
pip3 install opencv-python
pip3 install cython
pip3 install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI
```
Step 2:
Command to run the infernce and a scale score will be printed.
```
cd macro_benchmark/Mask_RCNN/sample
./run_test.sh
python cal_stat.py
````
## 4. Tensorflow NMT
Here we use the English-Vietnamese as the inference model.
IWSLT English-Vietnamese
Data: 133K examples, vocab=vocab.(vi|en), train=train.(vi|en) dev=tst2012.(vi|en), test=tst2013.(vi|en), download script.
The model is trained by 2-layer LSTMs of 512 units with bidirectional encoder (i.e., 1 bidirectional layers for the encoder), embedding dim is 512. LuongAttention (scale=True) is used together with dropout keep_prob of 0.8. All parameters are uniformly. They used SGD with learning rate 1.0 as follows: train for 12K steps (~ 12 epochs); after 8K steps, we start halving learning rate every 1K step.  
  
You will need to do the following to use the checkpoint and do inference:  
Step1: 
Download the source data first.  
```
nmt/scripts/download_iwslt15.sh /tmp/nmt_data
```
Step 2:
Then download the source code(from https://github.com/tensorflow/nmt)
```
git clone https://github.com/tensorflow/nmt.git
```
Step 3:
Run the command for inference. It will translate 1268 sentences.
```
python -m nmt.nmt --src=en --tgt=vi --ckpt=./envi_model_1/translate.ckpt --hparams_path=nmt/standard_hparams/iwslt15.json --out_dir=/tmp/envi --vocab_prefix=/tmp/nmt_data/vocab --inference_input_file=/tmp/nmt_data/tst2013.en --inference_output_file=/tmp/envi/output_infer --inference_ref_file=/tmp/nmt_data/tst2013.vi
```
batch size can be changed by infer_batch_size in file below. (You also need to delete /tmp/envi/hparams)
```
nmt/standard_hparams/iwslt15.json   
```
## 5. Tensorflow DeepSpeech  
Here we use a pre-trained English model as the inference model.  
Data: The model can only take audio file .wav format, with 16k freq and 16bit compression.  
Run the command for inference. It will translate two audio clips.
The DeepSpeech code is already downloaded(from https://github.com/mozilla/DeepSpeech)
Step 1:
```
pip3 install deepspeech-gpu
```
Step 2:
```
cd macro_benchmark/DeepSpeech
./0_download.sh
./1_run.sh
```
CUDA 8 needed. CUDA 9 is not supported yet.  
## 6. Tensorflow SSD  
The inference test is based on SSD tensorflow project (https://github.com/balancap/SSD-Tensorflow)
Here we use a pre-trained SSD model as the inference model weights.  
The sample test can only test on images from demo folder. We set 10 iterations with total detected images of 130. A scale score will be printed.  
```
cd macro_benchmark/ssd-tensorflow
./0_download.sh
./1_run.sh
python cal_stat.py
```

## 7. DeepInterestNetwork --- Alimama
The inference test is based on DeepInterestNetwork tensorflow project (https://github.com/zhougr1993/DeepInterestNetwork)
To avoid the training process, we used a pretained network checkpoint file to benchmark the inference performance.  
The source code is already downloaded.  
Step 1:Download dataset and preprocess. Download the amazon product dataset of electronics category, which has 498,196 products and 7,824,482 records, and extract it to raw_data/ folder.
```
cd macro_benchmark/DeepInterestNet/DeepInterestNetwork
mkdir raw_data/;
cd utils;
./0_download_raw.sh;
```
Step 2: Convert raw data to pandas dataframe, and remap categorical id.
```
pip3 install pandas
python 1_convert_pd.py;
python 2_remap_id.py
```
Step 4: Choose the method DIN and enter the folder to run the inference test
```
cd ../din
python build_dataset.py
python infer.py
```

## 8. SynNet --- Synthetic models
There are two models here. One is based on synthesize of 3 classic models which are alexnet, vgg16 and googlenet. The other model is based on *some* application running on Alibaba platform. 
The models are integrated into part 2). You can use the same command with the model name of synNet-3c or synNet.

## Suggestions
We are still keep working hard to developing our benchmark suites. We are welcome to any suggestions, contributions and improvements from anyone. Please do not hesitate to contact us if you want to involve. Thanks.
You could submit questions on Github or contact us through aimatrix@list.alibaba-inc.com

  
