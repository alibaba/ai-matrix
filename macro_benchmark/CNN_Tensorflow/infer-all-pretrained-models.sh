#!/bin/bash

models='googlenet resnet50 resnet152 densenet121'
batchs='32'

if [ ! -d ../pretrained_models/aimatrix-pretained-weights ]; then
    echo "Pretrained models do not exist!"
    exit
fi
mkdir results_infer_pretrained_models

for md in $models
do
    for batch in $batchs
    do
        echo "----------------------------------------------------------------"
        echo "Running $md with batch size of $batch"
        echo "----------------------------------------------------------------"
        start=`date +%s%N`
        python nvcnn.py --model=$md \
                    --batch_size=$batch \
                    --num_gpus=1 \
                    --data_dir=../pretrained_models/aimatrix-pretained-weights/CNN_Tensorflow/imagenet_validation_TF   \
                    --display_every=100  \
                    --log_dir=../pretrained_models/aimatrix-pretained-weights/CNN_Tensorflow/log_${md}_32-save/  \
                    --eval \
                    |& tee ./results_infer_pretrained_models/result_${md}_${batch}.txt
        end=`date +%s%N`
        total_time=$(((end-start)/1000000))
        total_images=49921 # fixed number as in imagenet validation data set
        system_performance=$((1000*$total_images/$total_time))
        echo "Total images is: $total_images" >> ./results_infer/result_${md}_${batch}.txt
        echo "Total running time in miliseconds is: $total_time" >> ./results_infer/result_${md}_${batch}.txt
        echo "System performance in images/second is: $system_performance" >> ./results_infer/result_${md}_${batch}.txt
    done
done

