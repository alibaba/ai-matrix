#!/bin/bash

NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"

export WORLD_SIZE=$NUM_ACCELERATORS

batchs='128 256 512'
MAX_ITER=3200
IMAGES_EPOCH=1281024

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
    python ./multiproc.py --nproc_per_node $NUM_ACCELERATORS \
        ./main.py --arch resnet50 -c fanin --label-smoothing 0.1 --fp16 --static-loss-scale 256 ../../dataset/imagenet --epochs 1 --training_only \
        --batch_size $batch --num_accelerators $NUM_ACCELERATORS \
        |& tee ./results/result_train_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    #total_images=$(($batch*$MAX_ITER*$NUM_ACCELERATORS))
    total_images=$(($IMAGES_EPOCH*$NUM_ACCELERATORS))
    system_performance=$((1000*$total_images/$total_time))
    echo "Total images is: $total_images" >> ./results/result_train_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_train_${batch}.txt
    #echo "Approximate accelerator performance in images/second is: $system_performance" >> ./results/result_train_${batch}.txt
    echo "System performance in images/second is: $system_performance" >> ./results/result_train_${batch}.txt
done

python process_results.py --train
