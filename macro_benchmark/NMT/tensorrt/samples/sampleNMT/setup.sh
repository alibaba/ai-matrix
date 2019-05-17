#!/bin/bash

make
cd data
cd nmt
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

