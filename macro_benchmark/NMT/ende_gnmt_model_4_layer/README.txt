These model checkpoints can reproduce the BLEU scores in the tensorflow NMT
Tutorial's benchmark section
(https://github.com/tensorflow/nmt#benchmarks). Each checkpoint zipped directory
includes model parameters and the inference graph.

You will need to do the following to use the checkpoint:

1. Clone the git tutorial repo:
```
git clone https://github.com/tensorflow/nmt/
```

2. Download the data.

For IWSLT English-Vietnamese:
```
nmt/scripts/download_iwslt15.sh /tmp/nmt_data
```

For WMT German-English:
```
nmt/scripts/wmt16_en_de.sh /tmp/wmt16
```

3. Run the Python code to load the checkpoint and make inference.

For IWSLT English-Vietnamese NMT Model with standard attention model:
```
python -m nmt.nmt \
    --src=en --tgt=vi \
    --ckpt=/path/to/envi_model/translate.ckpt \
    --hparams_path=nmt/standard_hparams/iwslt15.json \
    --out_dir=/tmp/envi \
    --vocab_prefix=/tmp/nmt_data/vocab \
    --inference_input_file=/tmp/nmt_data/tst2013.en \
    --inference_output_file=/tmp/envi/output_infer \
    --inference_ref_file=/tmp/nmt_data/tst2013.vi
```

For WMT German-English NMT Model with standard attention model:
```
python -m nmt.nmt \
    --src=de --tgt=en \
    --ckpt=/path/to/deen_model/translate.ckpt \
    --hparams_path=nmt/standard_hparams/wmt16.json \
    --out_dir=/tmp/deen \
    --vocab_prefix=/tmp/wmt16/vocab.bpe.32000 \
    --inference_input_file=/tmp/wmt16/newstest2015.tok.bpe.32000.de \
    --inference_output_file=/tmp/deen/output_infer \
    --inference_ref_file=/tmp/wmt16/newstest2015.tok.bpe.32000.en
```

For WMT German-English NMT with GNMT attention model:
```
python -m nmt.nmt \
    --src=de --tgt=en \
    --ckpt=/path/to/deen_gnmt_model_4_layer/translate.ckpt \
    --hparams_path=nmt/standard_hparams/wmt16_gnmt_4_layer.json \
    --out_dir=/tmp/deen_gnmt \
    --vocab_prefix=/tmp/wmt16/vocab.bpe.32000 \
    --inference_input_file=/tmp/wmt16/newstest2015.tok.bpe.32000.de \
    --inference_output_file=/tmp/deen_gnmt/output_infer \
    --inference_ref_file=/tmp/wmt16/newstest2015.tok.bpe.32000.en
```

For WMT English-German NMT with GNMT attention model:
```
python -m nmt.nmt \
    --src=en --tgt=de \
    --ckpt=/path/to/ende_gnmt_model_8_layer/translate.ckpt \
    --hparams_path=nmt/standard_hparams/wmt16_gnmt_8_layer.json \
    --out_dir=/tmp/ende_gnmt \
    --vocab_prefix=/tmp/wmt16/vocab.bpe.32000 \
    --inference_input_file=/tmp/wmt16/newstest2015.tok.bpe.32000.en \
    --inference_output_file=/tmp/ende_gnmt/output_infer \
    --inference_ref_file=/tmp/wmt16/newstest2015.tok.bpe.32000.de
```
