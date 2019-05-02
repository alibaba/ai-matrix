#!/bin/bash
# This script helps creating links to the datasets in another prepared AIMatrix tree
src_aim_root=$1
if [ -z $src_aim_root ]; then
    echo "Must input the root of another AIMatrix tree!"
    exit 1
fi
src_aim_root="$(cd "$(dirname "$1")"; pwd)/$(basename "$1")"
echo set up links to dataset under $src_aim_root

create_link() {
    srcdir=$1
    name=$2
    srcpath=$srcdir/$name
    if [ ! -e $srcpath ]; then
        echo "path not exist: $srcpath"
        exit 1
    fi
    echo linking $srcpath to `pwd`/$name
    rm -rf $name
    ln -s $srcpath $name
}


echo "##########################################"
echo "###     pretrained model weights       ###"
echo "##########################################"
cd ../pretrained_models
create_link "$src_aim_root/macro_benchmark/pretrained_models" "aimatrix-pretrained-weights"
cd ..

echo "##########################################"
echo "###        DeepInterest                ###"
echo "##########################################"
cd DeepInterest
create_link "$src_aim_root/macro_benchmark/DeepInterest" "raw_data"
create_link "$src_aim_root/macro_benchmark/DeepInterest" "din/dataset.pkl"
cd ..

echo "##########################################"
echo "###        Mask_RCNN                   ###"
echo "##########################################"
cd Mask_RCNN
create_link "$src_aim_root/macro_benchmark/Mask_RCNN" "mask_rcnn_coco.h5"
create_link "$src_aim_root/macro_benchmark/Mask_RCNN" "coco_dataset"
cd ..

echo "##########################################"
echo "###        NMT                         ###"
echo "##########################################"
cd NMT
create_link "$src_aim_root/macro_benchmark/NMT" "dataset"
cd ..

echo "##########################################"
echo "###        SSD                         ###"
echo "##########################################"
cd SSD
cd checkpoints
if [ ! -f "ssd_300_vgg.ckpt.data-00000-of-00001" ] || [ ! -f "ssd_300_vgg.ckpt.index" ]; then
    unzip ssd_300_vgg.ckpt.zip
else
    echo "Checkpoint already unzipped"
fi
cd ..
create_link "$src_aim_root/macro_benchmark/SSD" "VOC2007"
cd ..

echo "##########################################"
echo "###        DSSD                        ###"
echo "##########################################"
cd DSSD
create_link "$src_aim_root/macro_benchmark/DSSD" "VOC0712"
cd ..

echo "##########################################"
echo "###        DIEN                        ###"
echo "##########################################"
cd DIEN
./prepare_dataset.sh
cd ..
