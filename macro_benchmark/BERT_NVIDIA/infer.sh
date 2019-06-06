#!/bin/bash

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

batchs='1 2'

for batch in $batchs
do
    echo "----------------------------------------------------------------"
    echo "Running inference with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    ./scripts/finetune_inference_benchmark.sh squad fp32 true $batch checkpoints/model.ckpt |& tee ./results/result_infer_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    #total_images=$(($batch*$MAX_ITER*$NUM_ACCELERATORS))
    total_images=10833
    system_performance=$((1000*$total_images/$total_time))
    approximate_perf=`cat ./results/result_infer_${batch}.txt | grep -F 'Inference Performance' | awk -F' ' '{print $5}'`
    echo "Total sentences is: $total_images" >> ./results/result_infer_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_infer_${batch}.txt
    echo "Approximate inference accelerator performance in sentences/second is: $approximate_perf" >> ./results/result_infer_${batch}.txt
    echo "System performance in sentences/second is: $system_performance" >> ./results/result_infer_${batch}.txt
done

python process_results.py --infer
