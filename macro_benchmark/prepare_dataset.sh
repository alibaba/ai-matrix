#!/bin/bash

cd DeepInterest
./prepare_dataset.sh
cd ..

cd Mask_RCNN
./prepare_dataset.sh
cd ..

cd NMT
./prepare_dataset.sh
cd ..

cd SSD
./prepare_dataset.sh
cd ..

cd DSSD
./prepare_dataset.sh
cd ..

cd DIEN
./prepare_dataset.sh
cd ..
