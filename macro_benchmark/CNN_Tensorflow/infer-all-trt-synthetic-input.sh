#!/bin/bash

models='googlenet resnet50 resnet152 densenet121'
batchs='16 32 64'
precisions='fp32 fp16'
if [ ! -d ../pretrained_models/aimatrix-pretrained-weights ]; then
    echo "Pretrained models do not exist!"
    exit
fi 
mkdir results_infer_trt
num_batches=500
for md in $models
do
    for precision in $precisions 
    do
        for batch in $batchs
        do
            echo "----------------------------------------------------------------"
            echo "Running $md with batch size of $batch with precision $precision"
            echo "----------------------------------------------------------------"
            start=`date +%s%N`
            command="python nvcnn.py --model=$md \
                        --batch_size=$batch \
                        --num_gpus=1 \
                        --display_every=100  \
                        --eval \
                        --use_trt \
                        --trt_precision=$precision \
                        --num_batches=$num_batches \
                        --cache_path=../pretrained_models/aimatrix-pretrained-weights/CNN_Tensorflow/graphs_NHWC  \
                        |& tee ./results_infer_trt/result_${md}_${batch}_${precision}.txt"

            echo $command
            eval $command
            end=`date +%s%N`
            total_time=$(((end-start)/1000000))
            total_images=$(($batch * $num_batches))
            system_performance=$((1000*$total_images/$total_time))
            echo "Total images is: $total_images" >> ./results_infer_trt/result_${md}_${batch}_${precision}.txt
            echo "Total running time in miliseconds is: $total_time" >> ./results_infer_trt/result_${md}_${batch}_${precision}.txt
            echo "System performance in images/second is: $system_performance" >> ./results_infer_trt/result_${md}_${batch}_${precision}.txt
        done
    done 
done

