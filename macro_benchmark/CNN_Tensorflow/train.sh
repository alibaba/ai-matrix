#!/bin/bash
models=$1
batchs='16 32 64'
for batch in $batchs
do
    python nvcnn.py --model=$models \
                --batch_size=$batch \
                --num_gpus=1 \
                --num_batches=500   \
                --display_every=100 \
                --log_dir=./log_${models}_${batch}
done


