#!/bin/bash

pip3 install imgaug
pip3 install opencv-python
pip3 install 'keras==2.1.6'
pip3 uninstall -y scikit-image
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
sed -i 's/if type(resFile) == str or type(resFile) == unicode/if type(resFile) == str or type(resFile) == bytes/g' /usr/local/lib/python3.5/dist-packages/pycocotools-2.0-py3.5-linux-x86_64.egg/pycocotools/coco.py
