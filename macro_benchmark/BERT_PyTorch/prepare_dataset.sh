#!/usr/bin/env bash

export DATA_DIR=../dataset/bert
export WORK_DIR=.

python3 $WORK_DIR/data/bertPrep.py --action download --dataset google_pretrained_weights  # Includes vocab
python3 $WORK_DIR/data/bertPrep.py --action download --dataset squad

cd $DATA_DIR/download
if [ ! -d pytorch_pretrained_weights ]; then
	mkdir pytorch_pretrained_weights
fi

cd pytorch_pretrained_weights

if [ ! -d uncased_L-24_H-1024_A-16 ]; then
	mkdir uncased_L-24_H-1024_A-16
fi

cd uncased_L-24_H-1024_A-16

wget https://zenodo.org/record/3521072/files/bert_model.pt

cd ../../..

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi

cd ../../BERT_PyTorch
