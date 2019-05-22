#!/bin/bash

NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"

batchs='8 16 32'
MAX_ITER=1000

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

cd ../DSSD

export PYTHONPATH=`pwd`/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

for batch in $batchs
do
    rm -r models/ResNet-101/VOC0712
    echo "----------------------------------------------------------------"
    echo "Running training with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    python examples/ssd/ssd_pascal_resnet18_321.py \
            --batch_size=$batch \
            --num_accelerators=$NUM_ACCELERATORS \
            --max_iter=$MAX_ITER |& tee ../SSD_ResNet18_Caffe/results/result_train_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    total_images=$(($batch*$MAX_ITER*$NUM_ACCELERATORS))
    system_performance=$((1000*$total_images/$total_time))
    echo "Total images is: $total_images" >> ../SSD_ResNet18_Caffe/results/result_train_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ../SSD_ResNet18_Caffe/results/result_train_${batch}.txt
    echo "Approximate accelerator performance in images/second is: $system_performance" >> ../SSD_ResNet18_Caffe/results/result_train_${batch}.txt
    echo "System performance in images/second is: $system_performance" >> ../SSD_ResNet18_Caffe/results/result_train_${batch}.txt
done

cd -

python process_results.py --train
