#!/bin/bash

cd tool/convert_caffe_model
./run.sh
cd -

if [ ! -f "datasets/icdar_2015_incidental/ch4_test_images.zip" ]; then
    cd datasets/icdar_2015_incidental
    wget https://zenodo.org/record/3463706/files/ch4_test_images.zip
    cd ../..
fi

if [ ! -f "datasets/icdar_2015_incidental/ch4_training_images.zip" ]; then
    cd datasets/icdar_2015_incidental
    wget https://zenodo.org/record/3463706/files/ch4_training_images.zip
    cd ../..
fi

if [ ! -d "data" ]; then
	mkdir data
fi

cd datasets/icdar_2015_incidental

if [ ! -d "ch4_test_images" ]; then
	unzip ch4_test_images.zip
fi

if [ ! -d "ch4_training_images" ]; then
	unzip ch4_training_images.zip
fi

if [ ! -d "ch4_training_localization_transcription_gt" ]; then
	unzip ch4_training_localization_transcription_gt.zip
fi

if [ ! -d "Challenge4_Test_Task1_GT" ]; then
	unzip Challenge4_Test_Task1_GT.zip
fi

cd ..

if [ ! -f "SynthText.zip" ]; then
	wget http://www.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip
	unzip SynthText.zip
fi
cd ..

cd tool
python create_datasets.py
cd -

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi
