#!/bin/bash

apt-get update
apt-get install software-properties-common
add-apt-repository ppa:git-core/ppa
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git lfs install
pip3 install xlsxwriter
apt-get install bc
# won't go through if it is ubuntu
yum install bc 

echo "##########################################"
echo "### Set up SSD_Caffe                   ###"
echo "##########################################"
cd SSD_Caffe/tensorrt/sampleSSD
./setup.sh
cd ../../..
