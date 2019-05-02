#!/bin/bash

echo "##########################################"
echo "###        Download weights            ###"
echo "##########################################"
cd pretrained_models
./download_models.sh
cd ..

echo "##########################################"
echo "###        DeepInterest                ###"
echo "##########################################"
cd DeepInterest
./prepare_dataset.sh
cd ..

echo "##########################################"
echo "###        Mask_RCNN                   ###"
echo "##########################################"
cd Mask_RCNN
./prepare_dataset.sh
cd ..

echo "##########################################"
echo "###        NMT                         ###"
echo "##########################################"
cd NMT
./prepare_dataset.sh
cd ..

echo "##########################################"
echo "###        SSD                         ###"
echo "##########################################"
cd SSD
./prepare_dataset.sh
cd ..

echo "##########################################"
echo "###        DSSD                        ###"
echo "##########################################"
cd DSSD
./prepare_dataset.sh
cd ..

echo "##########################################"
echo "###        DIEN                        ###"
echo "##########################################"
cd DIEN
./prepare_dataset.sh
cd ..
