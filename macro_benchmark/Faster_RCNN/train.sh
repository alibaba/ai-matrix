#!/bin/bash

export PYTHONPATH=./lib/rpn
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

batchs='1'
MAX_ITER=10000

NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

for batch in $batchs
do
    echo "----------------------------------------------------------------"
    echo "Running training with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    ./experiments/scripts/faster_rcnn_end2end_train.sh 0 VGG16 pascal_voc |& tee ./results/result_train_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    total_images=$MAX_ITER
    system_performance=$((1000*$total_images/$total_time))
    echo "Total images is: $total_images" >> ./results/result_train_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_train_${batch}.txt
    echo "Approximate accelerator performance in images/second is: $system_performance" >> ./results/result_train_${batch}.txt
    echo "System performance in images/second is: $system_performance" >> ./results/result_train_${batch}.txt
done

python process_results.py --train
