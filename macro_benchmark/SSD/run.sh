#!/bin/bash

TRAIN_DIR=./logs/
DATASET_DIR="./VOC2007/trainval_tf/"
CHECKPOINT_PATH="./checkpoints/ssd_300_vgg.ckpt"

batchs='8 16 32'
num_of_steps=1000

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

### run training ###
for batch in $batchs
do
    rm -r $TRAIN_DIR
    echo "----------------------------------------------------------------"
    echo "Running training with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    python train_ssd_network.py \
        --train_dir=${TRAIN_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=pascalvoc_2007 \
        --dataset_split_name=train \
        --model_name=ssd_300_vgg \
        --checkpoint_path=${CHECKPOINT_PATH} \
        --save_summaries_secs=60 \
        --save_interval_secs=600 \
        --weight_decay=0.0005 \
        --optimizer=adam \
        --learning_rate=0.001 \
        --batch_size=$batch \
        --max_number_of_steps=$num_of_steps \
        |& tee ./results/result_train_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    total_images=$(($batch*$num_of_steps))
    system_performance=$((1000*$total_images/$total_time))
    echo "Total images is: $total_images" >> ./results/result_train_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_train_${batch}.txt
    echo "System performance in images/second is: $system_performance" >> ./results/result_train_${batch}.txt
done

python process_results.py --train

### run inference ###
EVAL_DIR="./logs"
DATASET_DIR="./VOC2007/test_tf/"

for batch in $batchs
do
    rm -r $EVAL_DIR
    echo "----------------------------------------------------------------"
    echo "Running inference with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    python eval_ssd_network.py \
        --eval_dir=${EVAL_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=pascalvoc_2007 \
        --dataset_split_name=test \
        --model_name=ssd_300_vgg \
        --checkpoint_path=${CHECKPOINT_PATH} \
        --batch_size=$batch \
        |& tee ./results/result_infer_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    total_images=4952
    system_performance=$((1000*$total_images/$total_time))
    echo "Total images is: $total_images" >> ./results/result_infer_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_infer_${batch}.txt
    echo "System performance in images/second is: $system_performance" >> ./results/result_infer_${batch}.txt
done

python process_results.py --infer
