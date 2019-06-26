#!/bin/bash

cd data
if [ ! -f "mjsynth.tar.gz" ]; then
	wget https://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
	tar -xvzf mjsynth.tar.gz
	mv mnt/ramdisk/max/90kDICT32px mjsynth
	rm -r mnt
fi
cd ..

PYTHONPATH=$PYTHONPATH:. python tools/write_tfrecords.py --dataset_dir data/mjsynth --save_dir data/mjsynth

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi
