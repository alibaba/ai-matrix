#!/bin/bash

NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

if [ ! -d output ]; then
    mkdir output
fi

batchs='1 2 4'

for batch in $batchs
do
    rm output/*
    echo "----------------------------------------------------------------"
    echo "Running training with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    ./scripts/finetune_train_benchmark.sh squad fp32 true $NUM_ACCELERATORS $batch |& tee ./results/result_train_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    # total_images=$(($batch*2000*$NUM_ACCELERATORS))
    total_images=8759
    system_performance=$((1000*$total_images/$total_time))
    approximate_perf=`cat ./results/result_train_${batch}.txt | grep -F 'Training Performance' | awk -F' ' '{print $5}'`
    echo "Total sentences is: $total_images" >> ./results/result_train_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_train_${batch}.txt
    echo "Approximate training accelerator performance in sentences/second is: $approximate_perf" >> ./results/result_train_${batch}.txt
    echo "System performance in sentences/second is: $system_performance" >> ./results/result_train_${batch}.txt
done

cd output
mv model.ckpt-*.data-00000-of-00001 model.ckpt.data-00000-of-00001
mv model.ckpt-*.index model.ckpt.index
mv model.ckpt-*.meta model.ckpt.meta
cd ..

if [ ! -d checkpoints ]; then
    mkdir checkpoints
fi
cp output/* checkpoints

python process_results.py --train
