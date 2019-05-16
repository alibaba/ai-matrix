#!/bin/bash

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

batchs='64 128 256'

for batch in $batchs
do
	echo "----------------------------------------------------------------"
	echo "Running inference with batch size of $batch"
	echo "----------------------------------------------------------------"
	rm -r deen_gnmt
	start=`date +%s%N`
	python -m nmt.nmt \
		--src=de --tgt=en \
		--ckpt=ende_gnmt_model_4_layer/translate.ckpt \
		--hparams_path=nmt/standard_hparams/wmt16_gnmt_4_layer.json \
		--out_dir=deen_gnmt \
		--vocab_prefix=dataset/wmt16_de_en/vocab.bpe.32000 \
		--inference_input_file=dataset/wmt16_de_en/newstest.tok.bpe.32000.de \
		--inference_output_file=deen_gnmt/output_infer \
		--inference_ref_file=dataset/wmt16_de_en/newstest.tok.bpe.32000.en \
		--infer_batch_size=$batch \
		|& tee ./results/result_infer_${batch}.txt
	end=`date +%s%N`
	total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    #total_images=$(($batch*$MAX_ITER*$NUM_ACCELERATORS))
    total_images=22191
    system_performance=$((1000*$total_images/$total_time))
    echo "Total sentences is: $total_images" >> ./results/result_infer_${batch}.txt
    echo "Total running time in miliseconds is: $total_time" >> ./results/result_infer_${batch}.txt
    echo "System performance in sentences/second is: $system_performance" >> ./results/result_infer_${batch}.txt
done

python process_results.py --infer
