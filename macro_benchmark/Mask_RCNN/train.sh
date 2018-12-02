#!/bin/bash

NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

cd samples/coco
start=`date +%s%N`
python3 coco.py train --dataset=../../coco_dataset --year=2014 --model=../../mask_rcnn_coco.h5 --num_accelerators=$NUM_ACCELERATORS |& tee ../../results/result_train.txt
end=`date +%s%N`
total_time=$(((end-start)/1000000))
total_images=3000*2*$NUM_ACCELERATORS
system_performance=`bc <<< "scale = 3; (1000*$total_images/$total_time)"`
echo "System time in miliseconds is: $total_time" >> ../../results/result_train.txt
echo "System performance in images/second is: $system_performance" >> ../../results/result_train.txt

cd ../..
python process_results.py --train
