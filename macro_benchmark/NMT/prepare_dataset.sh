#!/bin/bash

if [ ! -d "dataset" ]; then
	mkdir dataset
fi

cd dataset
#../nmt/scripts/wmt16_en_de.sh
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

