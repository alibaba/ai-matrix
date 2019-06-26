#!/bin/bash

cd data
if [ ! -f "mjsynth.tar.gz" ]; then
	wget https://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
	tar -xvzf mjsynth.tar.gz
	mv mnt/ramdisk/max/90kDICT32px mjsynth
	rm -r mnt
fi
cd ..

cd model
if [ ! -d "pretrained_model" ]; then
	mkdir pretrained_model
fi
cd pretrained_model

if [ ! -f "checkpoint" ]; then
	wget https://www.dropbox.com/sh/y4eaunamardibnd/AACFb77YH23N1Jw69lMqEoOxa/checkpoint
fi

if [ ! -f "shadownet.ckpt.data-00000-of-00001" ]; then
	wget https://www.dropbox.com/sh/y4eaunamardibnd/AADt_o5NmLdX8sUhNJOAGga9a/shadownet.ckpt.data-00000-of-00001
fi

if [ ! -f "shadownet.ckpt.index" ]; then
	wget https://www.dropbox.com/sh/y4eaunamardibnd/AAC3kcrbG7OfoYHOEapRoh-Ma/shadownet.ckpt.index
fi

if [ ! -f "shadownet.ckpt.meta" ]; then
	wget https://www.dropbox.com/sh/y4eaunamardibnd/AADD61vvTD0voXnvefB5lLQda/shadownet.ckpt.meta
fi

cd ../..

PYTHONPATH=$PYTHONPATH:. python tools/write_tfrecords.py --dataset_dir data/mjsynth --save_dir data/mjsynth

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi
