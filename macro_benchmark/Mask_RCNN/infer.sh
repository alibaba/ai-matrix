#!/bin/bash

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

cd samples/coco
start=`date +%s%N`
python3 coco.py evaluate --dataset=../../coco_dataset --year=2014 --model=../../mask_rcnn_coco.h5 |& tee ../../results/result_infer.txt
end=`date +%s%N`
total_time=$(((end-start)/1000000))
total_images=500
system_performance=`bc <<< "scale = 3; (1000*$total_images/$total_time)"`
echo "System time in miliseconds is: $total_time" >> ../../results/result_infer.txt
echo "System performance in images/second is: $system_performance" >> ../../results/result_infer.txt

cd ../..
python process_results.py --infer
