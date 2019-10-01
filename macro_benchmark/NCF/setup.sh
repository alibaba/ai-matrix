#!/bin/bash

pip3 install numpy==1.16.1

if [ ! -f "Data/train_data.npy" ]; then
        cd Data
    wget https://zenodo.org/api/files/950b6f8e-9676-462e-9704-6e065dde09f7/train_data.npy
    cd ..
fi

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi
