#!/bin/bash

NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"

export DATA_DIR=../dataset/bert
export WORK_DIR=.

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

if [ ! -d output ]; then
    mkdir output
fi

batchs='4 8 16'
MAX_STEPS=500

for batch in $batchs
do
    rm output/*
    echo "----------------------------------------------------------------"
    echo "Running training with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    ./scripts/run_squad.sh $batch $NUM_ACCELERATORS $MAX_STEPS |& tee ./results/result_train_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    total_images=$(($batch*$MAX_STEPS*$NUM_ACCELERATORS))
    system_performance=$((1000*$total_images/$total_time))
    approximate_perf=`cat ./results/result_train_${batch}.txt | grep -F 'training throughput:' | awk -F' ' '{print $3}'`
    echo "Total sentences is: $total_images" >> ./results/result_train_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_train_${batch}.txt
    echo "Approximate training accelerator performance in sentences/second is: $approximate_perf" >> ./results/result_train_${batch}.txt
    echo "System performance in sentences/second is: $system_performance" >> ./results/result_train_${batch}.txt
done

python process_results.py --train
