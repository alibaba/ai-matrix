#!/bin/bash

apt-get update
apt-get install software-properties-common
add-apt-repository ppa:git-core/ppa
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git lfs install

echo "##########################################"
echo "### Set up SSD_Caffe                   ###"
echo "##########################################"
cd SSD_Caffe/tensorrt/sampleSSD
./setup.sh
cd ../../..
