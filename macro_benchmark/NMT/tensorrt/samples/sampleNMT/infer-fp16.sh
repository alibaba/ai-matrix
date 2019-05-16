#!/bin/bash
batchs='64 128 256'
md="nmt"
if [ -d results_infer_fp16 ]; then
    mv results_infer_fp16 results_infer_fp16_$(date +%Y%m%d%H%M%S)
fi
mkdir results_infer_fp16

# fp16
for batch in $batchs
    do
        echo "----------------------------------------------------------------"
        echo "Running nmt+Tensor RT(fp16) with batch size of $batch"
        echo "----------------------------------------------------------------"
        start=`date +%s%N`
        command="../../bin/sample_nmt   \
		    --data_dir=data/nmt/deen \
		    --data_writer=benchmark \
                    --batch=$batch \
                    --fp16 \
                    |& tee ./results_infer_fp16/result_${md}_${batch}_fp16.txt"
        echo $command
        eval $command
        end=`date +%s%N`
        total_time=$(((end-start)/1000000))
        #total_images=$(($batch*$num_batches))
        total_images=22191
        system_performance=$((1000*$total_images/$total_time))
        echo "Total sentences is: $total_images" >> ./results_infer_fp16/result_${md}_${batch}_fp16.txt
        echo "Total running time in miliseconds is: $total_time" >> ./results_infer_fp16/result_${md}_${batch}_fp16.txt
        echo "System performance in sentences/second is: $system_performance" >> ./results_infer_fp16/result_${md}_${batch}_fp16.txt
done
python process_results.py --infer_trt_precision=fp16
