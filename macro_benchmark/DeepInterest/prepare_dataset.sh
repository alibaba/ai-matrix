#!/bin/bash

mkdir raw_data
cd utils
./0_download_raw.sh

cd ..
echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi

cd utils
python 1_convert_pd.py
python 2_remap_id.py
cd ../din
python build_dataset.py

