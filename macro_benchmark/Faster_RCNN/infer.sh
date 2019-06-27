#!/bin/bash

export PYTHONPATH=./lib/rpn
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

batchs='1'
NUM_IMAGES=4952

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
    echo "Running inference with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    ./experiments/scripts/faster_rcnn_end2end_infer.sh 0 VGG16 pascal_voc |& tee ./results/result_infer_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    #total_images=$(($batch*$MAX_ITER*$NUM_ACCELERATORS))
    total_images=$NUM_IMAGES
    system_performance=$((1000*$total_images/$total_time))
    echo "Total images is: $total_images" >> ./results/result_infer_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_infer_${batch}.txt
    echo "Approximate accelerator performance in images/second is: $system_performance" >> ./results/result_infer_${batch}.txt
    echo "System performance in images/second is: $system_performance" >> ./results/result_infer_${batch}.txt
done

python process_results.py --infer
