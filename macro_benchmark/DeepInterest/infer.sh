#!/bin/bash

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

batchs='1 32 64'

cd din

for batch in $batchs
do
	echo "----------------------------------------------------------------"
	echo "Running inference with batch size of $batch"
	echo "----------------------------------------------------------------"
	start=`date +%s%N`
	python infer.py --batch_size $batch |& tee ../results/result_infer_${batch}.txt
	end=`date +%s%N`
	total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
	echo "System performance in seconds is: $total_time" >> ../results/result_infer_${batch}.txt
done

cd ..
python process_results.py --infer
