#!/bin/bash

if [ ! -d "../../../VOCdevkit" ]; then
    cd ../../..
    ./prepare_dataset.sh
    cd -
fi

if [ ! -d "images" ]; then
    mkdir images
fi

cd ../../../VOCdevkit/VOC2007/JPEGImages

for image in *
do
    size=`identify -format "%wx%h" $image`
    if [ $size == "500x375" ]; then
        echo $image
        name=`echo $image | sed "s/.jpg/""/g"`
        convert $image ../../../tensorrt/samples/sampleFasterRCNN/images/$name.ppm
    fi
done

cd -

ls images | cat > images/list.txt

