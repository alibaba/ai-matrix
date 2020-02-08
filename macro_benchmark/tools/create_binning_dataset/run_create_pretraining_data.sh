output_dir="wiki_dupe2_lowercase"
mkdir $output_dir
python create_pretraining_data.py \
    --input_file=wiki_zh_2019_text_sentSeg/*/* \
    --output_file=$output_dir \
    --vocab_file=vocab.txt \
    --do_lower_case=True \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=2 |& tee log_wiki_sentSeg.txt
