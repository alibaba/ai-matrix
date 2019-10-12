#!/bin/bash

NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"

export WORLD_SIZE=$NUM_ACCELERATORS

batchs='32 64 128'
MAX_ITER=3200

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
    python -m torch.distributed.launch --nproc_per_node=$NUM_ACCELERATORS \
        train.py \
        --use-fp16 \
        --nhwc \
        --pad-input \
        --jit \
        --delay-allreduce \
        --opt-loss \
        --epochs 10 \
        --batch-size $batch \
        --max_iter $MAX_ITER \
        --warmup-factor 0 \
        --no-save \
        --threshold=0.23 \
        --data ../dataset/COCO2017 \
        |& tee ./results/result_train_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    total_images=$(($batch*$MAX_ITER*$NUM_ACCELERATORS))
    system_performance=$((1000*$total_images/$total_time))
    echo "Total images is: $total_images" >> ./results/result_train_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_train_${batch}.txt
    echo "Approximate accelerator performance in images/second is: $system_performance" >> ./results/result_train_${batch}.txt
    echo "System performance in images/second is: $system_performance" >> ./results/result_train_${batch}.txt
done

python process_results.py --train
