#!/bin/bash

if [ "`which apt`" == "/usr/bin/apt" ]; then
    apt update && apt-get install -y libsm6 libxext6
    apt-get install -y libxrender-dev libxext-dev libsm-dev
elif [ "`which yum`" == "/usr/bin/yum" ]; then
    yum -y install libXext libSM libXrender
else
    echo "unknown OS"
    exit
fi



pip3 install imgaug opencv-python keras==2.1.6 scikit-image cython
cd coco/PythonAPI
make
make install
python setup.py install
cd ../..
#sed -i 's/if type(resFile) == str or type(resFile) == unicode/if type(resFile) == str or type(resFile) == bytes/g' /usr/local/lib/python3.5/dist-packages/pycocotools-2.0-py3.5-linux-x86_64.egg/pycocotools/coco.py
