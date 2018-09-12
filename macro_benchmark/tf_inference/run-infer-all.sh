#!/bin/bash
models='alexnet googlenet vgg16 resnet50 resnet152 densenet121 statsNet'
batchs='16 32 64'

mkdir results
for md in $models
do
    for batch in $batchs
    do
        python nvcnn.py --model=$md \
                    --batch_size=$batch \
                    --num_gpus=1 \
                    --num_batches=500   \
                    --display_every=100  \
                    --log_dir=./log_${md}_${batch}  \
                    --eval  | tee ./results/result_${md}_${batch}.txt
    done
done
