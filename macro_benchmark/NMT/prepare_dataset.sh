#!/bin/bash

if [ ! -f "ende_gnmt_model_4_layer/translate.ckpt.data-00000-of-00001" ]; then
        cd ende_gnmt_model_4_layer
    wget https://zenodo.org/record/3463700/files/translate.ckpt.data-00000-of-00001
    cd ..
fi

if [ ! -d "dataset" ]; then
	mkdir dataset
fi

cd dataset
if [ ! -d "wmt16_de_en" ]; then
	../nmt/scripts/wmt16_en_de.sh
fi

cd wmt16_de_en
cat newstest2016.tok.bpe.32000.de > newstest.tok.bpe.32000.de
cat newstest2016.tok.bpe.32000.en > newstest.tok.bpe.32000.en
cat newstest2015.tok.bpe.32000.de >> newstest.tok.bpe.32000.de
cat newstest2015.tok.bpe.32000.en >> newstest.tok.bpe.32000.en
cat newstest2014.tok.bpe.32000.de >> newstest.tok.bpe.32000.de
cat newstest2014.tok.bpe.32000.en >> newstest.tok.bpe.32000.en
cat newstest2013.tok.bpe.32000.de >> newstest.tok.bpe.32000.de
cat newstest2013.tok.bpe.32000.en >> newstest.tok.bpe.32000.en
cat newstest2012.tok.bpe.32000.de >> newstest.tok.bpe.32000.de
cat newstest2012.tok.bpe.32000.en >> newstest.tok.bpe.32000.en
cat newstest2011.tok.bpe.32000.de >> newstest.tok.bpe.32000.de
cat newstest2011.tok.bpe.32000.en >> newstest.tok.bpe.32000.en
cat newstest2010.tok.bpe.32000.de >> newstest.tok.bpe.32000.de
cat newstest2010.tok.bpe.32000.en >> newstest.tok.bpe.32000.en
cat newstest2009.tok.bpe.32000.de >> newstest.tok.bpe.32000.de
cat newstest2009.tok.bpe.32000.en >> newstest.tok.bpe.32000.en
cd ../..

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi

