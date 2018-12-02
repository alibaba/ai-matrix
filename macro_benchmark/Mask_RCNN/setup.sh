#!/bin/bash

pip3 install imgaug
pip3 install opencv-python
pip3 install 'keras==2.1.6'
pip3 uninstall scikit-image
pip3 install scikit-image==0.13.1
pip3 install cython
apt update && apt install -y libsm6 libxext6
apt-get install -y libxrender-dev
yum install libXext libSM libXrender
cd coco/PythonAPI
make
make install
python setup.py install
cd ../..

