#!/bin/bash

pip3 install gdown
pip3 install -r requirement.txt
pip3 install opencv-python
apt update && apt install -y libsm6 libxext6
cd lib
make clean
make all
cd lib_kernel/lib_nms
./compile.sh
cd ../../..

