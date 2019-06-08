#!/bin/bash

NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"

TRAIN_EPOCH_SIZE=32561
EPOCHS_TRAIN=50

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

batchs='1024 2048 4096'

for batch in $batchs
do
	echo "----------------------------------------------------------------"
	echo "Running training with batch size of $batch"
	echo "----------------------------------------------------------------"
	rm -r census_model
	start=`date +%s%N`
	python census_main.py --train_epochs=$EPOCHS_TRAIN --epochs_between_evals=$EPOCHS_TRAIN --batch_size=$batch |& tee results/result_train_${batch}.txt
	end=`date +%s%N`
	total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    total_images=$(( $TRAIN_EPOCH_SIZE * $EPOCHS_TRAIN ))
    system_performance=$((1000*$total_images/$total_time))
    echo "Total recommendations: $total_images" >> results/result_train_${batch}.txt
    echo "System time in miliseconds is: $total_time" >> results/result_train_${batch}.txt
    echo "Approximate accelerator performance in recommendations/second is: $system_performance" >> results/result_train_${batch}.txt
    echo "System performance in recommendations/second is: $system_performance" >> results/result_train_${batch}.txt
done

python process_results.py --train
