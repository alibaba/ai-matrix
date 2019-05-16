#!/bin/bash

make
cd data
mkdir nmt
cd nmt
mkdir deen
cd deen

if [ ! -f "sampleNMT_weights.tar.gz" ]; then
	wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/models/sampleNMT_weights.tar.gz
	tar -xzf sampleNMT_weights.tar.gz
fi

if [ ! -d "weights" ]; then
	mv samples/nmt/deen/weights .
	rm -r samples
fi

cd ../../..

if [ ! -d "../../../dataset" ]; then
	mkdir ../../../dataset
	cd ../../../dataset
	../nmt/scripts/wmt16_en_de.sh
	cd ../tensorrt/samples/sampleNMT
fi

cd ../../../dataset/wmt16_de_en

cp newstest2015.tok.bpe.32000.de  newstest2015.tok.bpe.32000.en  vocab.bpe.32000.de  vocab.bpe.32000.en ../../tensorrt/samples/sampleNMT/data/nmt/deen
cat newstest2016.tok.bpe.32000.de >> ../../tensorrt/samples/sampleNMT/data/nmt/deen/newstest2015.tok.bpe.32000.de
cat newstest2016.tok.bpe.32000.en >> ../../tensorrt/samples/sampleNMT/data/nmt/deen/newstest2015.tok.bpe.32000.en
cat newstest2014.tok.bpe.32000.de >> ../../tensorrt/samples/sampleNMT/data/nmt/deen/newstest2015.tok.bpe.32000.de
cat newstest2014.tok.bpe.32000.en >> ../../tensorrt/samples/sampleNMT/data/nmt/deen/newstest2015.tok.bpe.32000.en
cat newstest2013.tok.bpe.32000.de >> ../../tensorrt/samples/sampleNMT/data/nmt/deen/newstest2015.tok.bpe.32000.de
cat newstest2013.tok.bpe.32000.en >> ../../tensorrt/samples/sampleNMT/data/nmt/deen/newstest2015.tok.bpe.32000.en
cat newstest2012.tok.bpe.32000.de >> ../../tensorrt/samples/sampleNMT/data/nmt/deen/newstest2015.tok.bpe.32000.de
cat newstest2012.tok.bpe.32000.en >> ../../tensorrt/samples/sampleNMT/data/nmt/deen/newstest2015.tok.bpe.32000.en
cat newstest2011.tok.bpe.32000.de >> ../../tensorrt/samples/sampleNMT/data/nmt/deen/newstest2015.tok.bpe.32000.de
cat newstest2011.tok.bpe.32000.en >> ../../tensorrt/samples/sampleNMT/data/nmt/deen/newstest2015.tok.bpe.32000.en
cat newstest2010.tok.bpe.32000.de >> ../../tensorrt/samples/sampleNMT/data/nmt/deen/newstest2015.tok.bpe.32000.de
cat newstest2010.tok.bpe.32000.en >> ../../tensorrt/samples/sampleNMT/data/nmt/deen/newstest2015.tok.bpe.32000.en
cat newstest2009.tok.bpe.32000.de >> ../../tensorrt/samples/sampleNMT/data/nmt/deen/newstest2015.tok.bpe.32000.de
cat newstest2009.tok.bpe.32000.en >> ../../tensorrt/samples/sampleNMT/data/nmt/deen/newstest2015.tok.bpe.32000.en

cd ../..

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi

