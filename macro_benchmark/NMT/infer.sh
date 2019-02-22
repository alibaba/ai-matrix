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
	rm -r envi
	start=`date +%s%N`
	python -m nmt.nmt \
		--src=en --tgt=vi \
		--ckpt=./envi_model_1/translate.ckpt \
		--hparams_path=nmt/standard_hparams/iwslt15.json \
		--out_dir=envi \
		--vocab_prefix=dataset/en_vi/vocab \
		--inference_input_file=dataset/en_vi/tst2013.en \
		--inference_output_file=envi/output_infer \
		--inference_ref_file=dataset/en_vi/tst2013.vi \
		--infer_batch_size=$batch \
		|& tee ./results/result_infer_${batch}.txt
	end=`date +%s%N`
    total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    echo "System performance in seconds is: $total_time" >> ./results/result_infer_${batch}.txt
done

python process_results.py --infer
