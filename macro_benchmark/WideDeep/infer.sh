#!/bin/bash

INFER_EPOCH_SIZE=16281
EPOCHS_INFER=1000

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

batchs='1024 2048 4096'

for batch in $batchs
do
	echo "----------------------------------------------------------------"
	echo "Running inference with batch size of $batch"
	echo "----------------------------------------------------------------"
	start=`date +%s%N`
	python census_main.py --train_epochs=$EPOCHS_INFER --epochs_between_evals=$EPOCHS_INFER --batch_size=$batch --infer=True |& tee results/result_infer_${batch}.txt
	end=`date +%s%N`
	total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    total_images=$(( $INFER_EPOCH_SIZE * $EPOCHS_INFER ))
    system_performance=$((1000*$total_images/$total_time))
    echo "Total recommendations: $total_images" >> results/result_infer_${batch}.txt
    echo "System time in miliseconds is: $total_time" >> results/result_infer_${batch}.txt
    echo "Approximate accelerator performance in recommendations/second is: $system_performance" >> results/result_infer_${batch}.txt
    echo "System performance in recommendations/second is: $system_performance" >> results/result_infer_${batch}.txt
done

python process_results.py --infer
