#!/bin/bash

pip3 install gdown
pip3 install -r requirement.txt
pip3 install opencv-python
apt update && apt install -y libsm6 libxext6


pip3 install imgaug opencv-python keras==2.1.6 scikit-image cython
apt-get install python3-tk
cd coco/PythonAPI
make
make install
python setup.py install
cd ../..
sed -i 's/if type(resFile) == str or type(resFile) == unicode/if type(resFile) == str or type(resFile) == bytes/g' /usr/local/lib/python3.5/dist-packages/pycocotools/coco.py
