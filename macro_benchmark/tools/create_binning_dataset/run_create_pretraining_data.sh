output_dir="english_128_16/$1"
mkdir $output_dir
python create_pretraining_data.py \
    --input_file="$DATA_DIR/sharded_training_shards_256_test_shards_256_fraction_0.2/books_wiki_en_corpus/training/$1/*" \
    --output_file=$output_dir \
    --vocab_file="$DATA_DIR/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt" \
    --do_lower_case=True \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --interval=16 \
    --dupe_factor=5 |& tee log_wiki_sentSeg.txt
