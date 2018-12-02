#!/bin/bash
models=$1
batchs='16 32 64'
NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"
for batch in $batchs
do
    python nvcnn.py --model=$models \
                --batch_size=$batch \
                --num_gpus=$NUM_ACCELERATORS \
                --num_batches=500   \
                --display_every=100 \
                --log_dir=./log_${models}_${batch}
done


