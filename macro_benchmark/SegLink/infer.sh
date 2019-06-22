#!/bin/bash

batchs='4 8 16'
NUM_IMAGES=5000

NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"

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
    ./manage.py test exp/sgd test_ic15 $batch $NUM_IMAGES |& tee ./results/result_infer_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    total_images=$NUM_IMAGES
    system_performance=$((1000*$total_images/$total_time))
    echo "Total images is: $total_images" >> ./results/result_infer_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_infer_${batch}.txt
    echo "Approximate accelerator performance in images/second is: $system_performance" >> ./results/result_infer_${batch}.txt
    echo "System performance in images/second is: $system_performance" >> ./results/result_infer_${batch}.txt
done

python process_results.py --infer
