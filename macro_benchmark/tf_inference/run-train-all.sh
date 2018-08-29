#!/bin/bash

models='alexnet googlenet vgg16 resnet50 resnet152 densenet121 synNet'
batchs='16 32 64'
for md in $models
do
echo $md
    for batch in $batchs
    do
        python nvcnn.py --model=$md \
                    --batch_size=$batch \
                    --num_gpus=1 \
                    --num_batches=200   \
                    --display_every=100 \
                    --log_dir=./log_${md}_${batch}  
    done
done 

