#!/bin/bash

models='googlenet resnet50 resnet152 densenet121'
batchs='16 32 64'
num_batches=500

if [ -d results_infer ]; then
    mv results_infer results_infer_$(date +%Y%m%d%H%M%S)
fi
mkdir results_infer

for md in $models
do
    for batch in $batchs
    do
        echo "----------------------------------------------------------------"
        echo "Running $md with batch size of $batch"
        echo "----------------------------------------------------------------"
        start=`date +%s%N`
        command="python nvcnn.py --model=$md \
                    --batch_size=$batch \
                    --num_gpus=1 \
                    --num_batches=$num_batches   \
                    --display_every=100  \
                    --log_dir=../pretrained_models/aimatrix-pretrained-weights/CNN_Tensorflow/log_${md}_32-save \
                    --eval \
                    |& tee ./results_infer/result_${md}_${batch}.txt"
        echo $command
        eval $command
        end=`date +%s%N`
        total_time=$(((end-start)/1000000))
        total_images=$(($batch*$num_batches))
        system_performance=$((1000*$total_images/$total_time))
        echo "Total images is: $total_images" >> ./results_infer/result_${md}_${batch}.txt
        echo "Total running time in miliseconds is: $total_time" >> ./results_infer/result_${md}_${batch}.txt
        echo "System performance in images/second is: $system_performance" >> ./results_infer/result_${md}_${batch}.txt
    done
done

python process_results.py --infer
