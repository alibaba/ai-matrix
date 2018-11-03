### Tensorflow CNN  

hardware: CPU+GPU =  Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GH  + Nvidia V100
software: TF1.10 +CUDA9+cudnn7.1  

| Model           | Batch size | Img/sec (FP32) +- std |
|-----------------|----------|----------------|
| alexnet         | 16         | 5947 =- 24           |
| ​                | 32         | 7800 +- 38           |
| ​                | 64         | 9103 +- 37           |
| googlenet       | 16         | 1706 +- 1           |
| ​                | 32         | 2202 +- 4           |
| ​                | 64         | 2561 +- 5           |
| vgg16           | 16         | 626 +- 1            |
| ​                | 32         | 672 +- 0            |
| ​                | 64         | 685 +- 0            |
| resnet50        | 16         | 949 +- 2            |
| ​                | 32         | 1079 +- 3           |
| ​                | 64         | 1156 +- 1          |
| resnet152       | 16         | 377 +- 1            |
| ​                | 32         | 435 +- 1            |
| ​                | 64         | 458 +- 0           |
| densenet121_k32 | 16         | 839 +- 2            |
| ​                | 32         | 1049 +- 2          |
| ​                | 64         | 1181 +- 1           |
| synNet           | 16         | 321 +- 0           |
| ​                | 32         | 377 +- 0           |
| ​                | 64         | 409 +- 0           |

hardware: CPU+GPU =  Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GH  + Nvidia P4
software: TF1.10 +CUDA9+cudnn7.1  

| Model           | Batch size | Img/sec (FP32) +- std|
|-----------------|----------|----------------|
| alexnet         | 16         | 1826 +- 6           |
| ​                | 32         | 2283 +- 5           |
| ​                | 64         | 2551 +- 4           |
| googlenet       | 16         | 576 +- 1            |
| ​                | 32         | 648 +- 1            |
| ​                | 64         | 682 +- 1           |
| vgg16           | 16         | 174 +- 0            |
| ​                | 32         | 182 +- 1           |
| ​                | 64         | 184 +- 2            |
| resnet50        | 16         | 258 +- 0            |
| ​                | 32         | 280 +- 0           |
| ​                | 64         | 297 +- 0            |
| resnet152       | 16         | 108 +- 0            |
| ​                | 32         | 117 +- 0            |
| ​                | 64         | 125 +- 0           |
| densenet121_k32 | 16         | 261 +- 0            |
| ​                | 32         | 274 +- 0           |
| ​                | 64         | 282 +- 0           |
| synNet           | 16         | 95 +- 0           |  
| ​                | 32         | 105 +- 0           |
| ​                | 64         | 111 +- 0           |

---
### Mask RCNN  

The Mask RCNN inference performance results are shown below. The resnet101 is used as backbone in this test framework. 28 images are tested in the inference situation. Each image is duplicated to make total batch size = n*28.  

|    CPU+GPU                                            |    Batch size    |    Time (s)    |
|-------------------------------------------------------|------------------|----------------|
| Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GH, Nvidia V100 | 4                | 40.2           |
|                                                       | 8                | 73.6           |
|                                                       | 16               | 136.7          |
| Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GH, Nvidia P4   | 4                | 63.7           |
|                                                       | 8                | 119.6          |
|                                                       | 16               | --             |
*-- means out of memory  

---
### Tensorflow SSD  

A pre-trained SSD model is loaded as the inference model weights. The sample test can only test on images from demo folder. We set 10 iterations with total detected images of 130.  

| CPU+GPU                                        | test       | time (s) |
|------------------------------------------------|------------|----------|
| Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GH +V100 | 130 images | 5.83     |
| Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GH + P4  | 130 images | 7.41     |
  
---
### Tensorflow NMT  

Here we use the English-Vietnamese as the inference model which is trained on IWSLT English-Vietnamese dataset. The inference test is on a dataset with 1268 sentences with difference batch size.  

| CPU+GPU                                               | Batch size | Time (s) |
|-------------------------------------------------------|------------|----------|
| Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GH, Nvidia V100 | 16         | 14.9     |
|                                                       | 32         | 10.4     |
|                                                       | 64         | 7.4      |
| Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GH, Nvidia P4   | 16         | 22.69    |
|                                                       | 32         | 18.63    |
|                                                       | 64         | 16.26    |
    
---
### Tensorflow DeepSpeech  

The model can only take audio file .wav format, with 16k freq and 16bit compression. We test on 2 files with data show below:  

| CPU+GPU                                        | Test case                    | Time (s) |
|------------------------------------------------|------------------------------|----------|
| Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GH +V100 | test1 for 31.7s audio file   | 10.92    |
|                                                | test2 for 12.195s audio file | 6.11     |
| Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GH + P4  | test1 for 31.7s audio file   | 15.05    |
|                                                | test2 for 12.195s audio file | 7.1      |
  
---
### DeepInterestNetwork  

The inference test is based on DeepInterestNetwork tensorflow project.  We used a pretained network (trained for a few epoches) to benchmark the inference performance.  

| CPU+GPU                                        | Batch size | Perf (recommendation processed/sec) |
|------------------------------------------------|------------|-------------------------------------|
| Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GH +V100 | 256        | 68129                               |
|                                                | 512        | 88441                               |
|                                                | 1024       | 97492                               |
| Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GH + P4  | 256        | 48302                               |
|                                                | 512        | 49491                               |
|                                                | 1024       | 44911                               |

