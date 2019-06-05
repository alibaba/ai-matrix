#!/bin/bash

NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

batchs='64'

for batch in $batchs
do
    echo "----------------------------------------------------------------"
    echo "Running training with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    python run_classifier.py \
	--task_name=MNLI \
	--do_train=true \
	--do_eval=true \
	--do_lower_case=False \
	--data_dir=glue_data/MNLI \
	--vocab_file=bert_base/multi_cased_L-12_H-768_A-12/vocab.txt \
	--bert_config_file=bert_base/multi_cased_L-12_H-768_A-12/bert_config.json \
	--init_checkpoint=bert_base/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
	--max_seq_length=128 \
	--train_batch_size=32 \
	--learning_rate=2e-5 \
	--num_train_epochs=1.0 \
	--output_dir=mrpc_output \
	|& tee ./results/result_train_${batch}.txt
    end=`date +%s%N`
    total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    total_images=$(($batch*2000*$NUM_ACCELERATORS))
    system_performance=$((1000*$total_images/$total_time))
    echo "Total sentences is: $total_images" >> ./results/result_train_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_train_${batch}.txt
    echo "System performance in sentences/second is: $system_performance" >> ./results/result_train_${batch}.txt
done

python process_results.py --train
