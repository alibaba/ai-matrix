#!/bin/bash

batchs='128 256 512'
MAX_ITER=2000

NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

for batch in $batchs
do
    echo "----------------------------------------------------------------"
    echo "Running training with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    PYTHONPATH=$PYTHONPATH:. python tools/train_shadownet.py \
        --dataset_dir data/mjsynth \
        --char_dict_path data/char_dict/char_dict.json \
        --ord_map_dict_path data/char_dict/ord_map.json \
        --batch_size $batch \
        --num_iters $MAX_ITER \
        |& tee ./results/result_train_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    total_images=$(($batch*$MAX_ITER*$NUM_ACCELERATORS))
    system_performance=$((1000*$total_images/$total_time))
    echo "Total images is: $total_images" >> ./results/result_train_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_train_${batch}.txt
    echo "Approximate accelerator performance in images/second is: $system_performance" >> ./results/result_train_${batch}.txt
    echo "System performance in images/second is: $system_performance" >> ./results/result_train_${batch}.txt
done

python process_results.py --train

MAX_ITER=1600

for batch in $batchs
do
    echo "----------------------------------------------------------------"
    echo "Running inference with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    PYTHONPATH=$PYTHONPATH:. python tools/evaluate_shadownet.py \
        --dataset_dir data/mjsynth \
        --weights_path model/crnn_syn90k/shadownet.ckpt-2000 \
        --char_dict_path data/char_dict/char_dict.json \
        --ord_map_dict_path data/char_dict/ord_map.json \
        --num_iters $MAX_ITER \
        --batch_size $batch \
        --process_all 1 \
        --visualize 0 \
        |& tee ./results/result_infer_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    total_images=$(($batch*$MAX_ITER*$NUM_ACCELERATORS))
    system_performance=$((1000*$total_images/$total_time))
    echo "Total images is: $total_images" >> ./results/result_infer_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_infer_${batch}.txt
    echo "Approximate accelerator performance in images/second is: $system_performance" >> ./results/result_infer_${batch}.txt
    echo "System performance in images/second is: $system_performance" >> ./results/result_infer_${batch}.txt
done

python process_results.py --infer

