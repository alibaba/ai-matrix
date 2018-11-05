#!/bin/bash

mkdir raw_data
cd utils
./0_download_raw.sh
python 1_convert_pd.py
python 2_remap_id.py
cd ../din
python build_dataset.py

