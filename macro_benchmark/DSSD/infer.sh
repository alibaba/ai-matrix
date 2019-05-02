#!/bin/bash

export PYTHONPATH=`pwd`/python

batchs='4'

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
    ./build/tools/caffe test \
    -model="./models/ResNet-101/VOC0712/DSSD_VOC07_FT_321x321/test.prototxt" \
    -weights="./models/ResNet-101/VOC0712/DSSD_VOC07_FT_321x321/ResNet-101_VOC0712_DSSD_VOC07_FT_321x321_iter_1000.caffemodel" \
    -gpu=0 -iterations=1000 \
    |& tee ./results/result_infer_${batch}.txt
    end=`date +%s%N`
    #total_time=$(((end-start)/1000000))
    total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    #total_images=$(($batch*$num_of_steps*$NUM_ACCELERATORS))
    #system_performance=$((1000*$total_images/$total_time))
    #echo "Total images is: $total_images" >> ./results/result_train_${batch}.txt
    #echo "Total running time in miliseconds is: $total_time" >> ./results/result_train_${batch}.txt
    echo "Approximate accelerator time in seconds is: $total_time" >> ./results/result_infer_${batch}.txt
    echo "System performance in seconds is: $total_time" >> ./results/result_infer_${batch}.txt
done

python process_results.py --infer
