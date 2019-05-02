#!/bin/bash

models='googlenet resnet50 resnet152 densenet121 synNet'
batchs='64'
num_batches=2000
NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"

if [ -d results_train_pressure ]; then
    mv results_train_pressure results_train_pressure_$(date +%Y%m%d%H%M%S)
fi
mkdir results_train_pressure

rm -r log_*

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
                    --num_gpus=$NUM_ACCELERATORS \
                    --num_batches=$num_batches   \
                    --display_every=100 \
                    --log_dir=./log_${md}_${batch} \
                    |& tee ./results_train_pressure/result_${md}_${batch}.txt
        end=`date +%s%N`
        total_time=$(((end-start)/1000000))
        total_images=$(($batch*$num_batches*$NUM_ACCELERATORS))
        system_performance=$((1000*$total_images/$total_time))
        echo "Total images is: $total_images" >> ./results_train_pressure/result_${md}_${batch}.txt
        echo "Total running time in miliseconds is: $total_time" >> ./results_train_pressure/result_${md}_${batch}.txt
        echo "System performance in images/second is: $system_performance" >> ./results_train_pressure/result_${md}_${batch}.txt
    done
done

