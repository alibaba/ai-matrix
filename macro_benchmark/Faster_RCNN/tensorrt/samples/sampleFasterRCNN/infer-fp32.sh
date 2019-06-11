#!/bin/bash
batchs='2 4 6'
md="faster_rcnn"
if [ -d results_infer_fp32 ]; then
    mv results_infer_fp32 results_infer_fp32_$(date +%Y%m%d%H%M%S)
fi
mkdir results_infer_fp32
# fp32
for batch in $batchs
    do
        echo "----------------------------------------------------------------"
        echo "Running faster rcnn+Tensor RT(fp32) with batch size of $batch"
        echo "----------------------------------------------------------------"
        start=`date +%s%N`
        command="../../bin/sample_fasterRCNN --batch_size=$batch \
                    |& tee ./results_infer_fp32/result_${md}_${batch}_fp32.txt"
        echo $command
        eval $command
        end=`date +%s%N`
        total_time=$(((end-start)/1000000))
        total_images=4232
        system_performance=$((1000*$total_images/$total_time))
        echo "Total images is: $total_images" >> ./results_infer_fp32/result_${md}_${batch}_fp32.txt
        echo "Total running time in miliseconds is: $total_time" >> ./results_infer_fp32/result_${md}_${batch}_fp32.txt
        echo "Approximate accelerator performance in images/second is: $system_performance" >> ./results_infer_fp32/result_${md}_${batch}_fp32.txt
        echo "System performance in images/second is: $system_performance" >> ./results_infer_fp32/result_${md}_${batch}_fp32.txt
done
python process_results.py --infer_trt_precision=fp32
