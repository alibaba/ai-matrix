#!/bin/bash

echo "##########################################"
echo "###        Download weights            ###"
echo "##########################################"
cd pretrained_models
./download_models.sh
cd ..

echo "##########################################"
echo "###        Download dataset            ###"
echo "##########################################"
cd dataset
./prepare_dataset.sh
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
echo "###        DSSD                        ###"
echo "##########################################"
cd DSSD
./prepare_dataset.sh
cd ..

# echo "##########################################"
# echo "###        SSD_VGG16_Caffe             ###"
# echo "##########################################"
# cd SSD_VGG16_Caffe
# ./prepare_dataset.sh
# cd ..

echo "##########################################"
echo "###        SSD_ResNet18_Caffe          ###"
echo "##########################################"
cd SSD_ResNet18_Caffe
./prepare_dataset.sh
cd ..

echo "##########################################"
echo "###        DIEN                        ###"
echo "##########################################"
cd DIEN
./prepare_dataset.sh
cd ..

echo "##########################################"
echo "###        BERT_NVIDIA                 ###"
echo "##########################################"
cd BERT_NVIDIA
./prepare_dataset.sh
cd ..

echo "##########################################"
echo "###        Faster_RCNN                 ###"
echo "##########################################"
cd Faster_RCNN
./prepare_dataset.sh
cd ..

echo "##########################################"
echo "###        WideDeep                    ###"
echo "##########################################"
cd WideDeep
./prepare_dataset.sh
cd ..

echo "##########################################"
echo "###        CPN                         ###"
echo "##########################################"
cd CPN
./prepare_dataset.sh
cd ..

echo "##########################################"
echo "###        SegLink                     ###"
echo "##########################################"
cd SegLink
./prepare_dataset.sh
cd ..

echo "##########################################"
echo "###        CRNN                        ###"
echo "##########################################"
cd CRNN
./prepare_dataset.sh
cd ..
