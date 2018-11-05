#!/bin/bash

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

cd samples/coco
start=`date +%s%N`
python3 coco.py evaluate --dataset=../../coco_dataset --year=2014 --model=../../mask_rcnn_coco.h5 |& tee ../../results/result_infer.txt
end=`date +%s%N`
total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
echo "System performance in seconds is: $total_time" >> ../../results/result_infer.txt

cd ../..
python process_results.py --infer
