#!/bin/bash

cd CNN_Tensorflow
./setup.sh
cd ..

cd DeepInterest
./setup.sh
cd ..

cd Mask_RCNN
./setup.sh
cd ..

pip3 install xlsxwriter
