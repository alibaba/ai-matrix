#!/bin/bash

start=`date +%s%N`
start_date=`date`

echo "##########################################"
echo "### Running CNN_Tensorflow             ###"
echo "##########################################"
cd CNN_Tensorflow
./train-all.sh
cd ..

echo "##########################################"
echo "### Running DeepInterest               ###"
echo "##########################################"
cd DeepInterest
./train.sh
cd ..

echo "##########################################"
echo "### Running Mask_RCNN                  ###"
echo "##########################################"
cd Mask_RCNN
./train.sh
cd ..

echo "##########################################"
echo "### Running NMT                        ###"
echo "##########################################"
cd NMT
./train.sh
cd ..

echo "##########################################"
echo "### Running SSD_VGG16_Caffe            ###"
echo "##########################################"
cd SSD_VGG16_Caffe
./train.sh
cd ..

echo "##########################################"
echo "### Running SSD_ResNet18_Caffe         ###"
echo "##########################################"
cd SSD_ResNet18_Caffe
./train.sh
cd ..

echo "##########################################"
echo "### Running DSSD                       ###"
echo "##########################################"
cd DSSD
./train.sh
cd ..

echo "##########################################"
echo "### Running NCF                        ###"
echo "##########################################"
cd NCF
./train.sh
cd ..

echo "##########################################"
echo "### Running DIEN                       ###"
echo "##########################################"
cd DIEN
./train.sh
cd ..

./process_results.sh

end=`date +%s%N`
end_date=`date`
total_time=`bc <<< "scale = 0; ($end-$start)/1000000000"`
total_hours=`bc <<< "scale = 0; ${total_time}/3600"`
total_minutes=`bc <<< "sale = 0; (${total_time}%3600)/60"`
total_seconds=`bc <<< "scale = 0; ${total_time}%60"`
echo "Running started at ${start_date}"
echo "          ended at ${end_date}"
echo "Total running time is ${total_hours}h ${total_minutes}m ${total_seconds}s"
