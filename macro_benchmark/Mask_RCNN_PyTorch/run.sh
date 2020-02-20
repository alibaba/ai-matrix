#!/bin/bash

NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

batchs='2 4 8'
MAX_STEPS=2000

export WORLD_SIZE=1
export DATA_DIR=`pwd`"/../dataset/COCO2017"

for batch in $batchs
do
    echo "----------------------------------------------------------------"
    echo "Running training with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    python -m torch.distributed.launch --nproc_per_node=$NUM_ACCELERATORS \
        tools/train_mlperf.py \
        --config-file 'configs/e2e_mask_rcnn_R_50_FPN_1x.yaml' \
        DTYPE 'float16' \
        PATHS_CATALOG 'maskrcnn_benchmark/config/paths_catalog_dbcluster.py' \
        MODEL.WEIGHT './R-50.pkl' \
        DISABLE_REDUCED_LOGGING True \
        SOLVER.MAX_ITER $MAX_STEPS \
        SOLVER.IMS_PER_BATCH $batch \
        |& tee ./results/result_train_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    total_images=$(($batch*$MAX_STEPS*$NUM_ACCELERATORS))
    system_performance=$((1000*$total_images/$total_time))
    approximate_perf=`cat ./results/result_train_${batch}.txt | grep -F 'MLPERF METRIC THROUGHPUT:' | awk -F' ' '{print $6}'`
    echo "Total sentences is: $total_images" >> ./results/result_train_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_train_${batch}.txt
    echo "Approximate training accelerator performance in sentences/second is: $approximate_perf" >> ./results/result_train_${batch}.txt
    echo "System performance in sentences/second is: $system_performance" >> ./results/result_train_${batch}.txt
done

python process_results.py --train
