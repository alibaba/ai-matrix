#!/bin/bash

start=`date +%s%N`
start_date=`date`

echo "##########################################"
echo "### Running CNN_Tensorflow             ###"
echo "##########################################"
cd CNN_Tensorflow
./run.sh
cd ..

echo "##########################################"
echo "### Running DeepInterest               ###"
echo "##########################################"
cd DeepInterest
./run.sh
cd ..

echo "##########################################"
echo "### Running Mask_RCNN                  ###"
echo "##########################################"
cd Mask_RCNN
./run.sh
cd ..

echo "##########################################"
echo "### Running NMT                        ###"
echo "##########################################"
cd NMT
./run.sh
cd ..

# echo "##########################################"
# echo "### Running SSD_Tensorflow             ###"
# echo "##########################################"
# cd SSD_Tensorflow
# ./run.sh
# cd ..

# echo "##########################################"
# echo "### Running SSD_VGG16_Caffe            ###"
# echo "##########################################"
# cd SSD_VGG16_Caffe
# ./run.sh
# cd ..

echo "##########################################"
echo "### Running SSD_ResNet34_PyTorch       ###"
echo "##########################################"
cd SSD_ResNet34_PyTorch
./run.sh
cd ..

echo "##########################################"
echo "### Running SSD_ResNet18_Caffe         ###"
echo "##########################################"
cd SSD_ResNet18_Caffe
./run.sh
cd ..

echo "##########################################"
echo "### Running DSSD                       ###"
echo "##########################################"
cd DSSD
./run.sh
cd ..

echo "##########################################"
echo "### Running DIEN                       ###"
echo "##########################################"
cd DIEN
./run.sh
cd ..

echo "##########################################"
echo "### Running NCF                        ###"
echo "##########################################"
cd NCF
./run.sh
cd ..

echo "##########################################"
echo "### Running BERT_NVIDIA                ###"
echo "##########################################"
cd BERT_NVIDIA
./run.sh
cd ..

echo "##########################################"
echo "### Running Faster_RCNN                ###"
echo "##########################################"
cd Faster_RCNN
./run.sh
cd ..

echo "##########################################"
echo "### Running WideDeep                   ###"
echo "##########################################"
cd WideDeep
./run.sh
cd ..

echo "##########################################"
echo "### Running CPN                        ###"
echo "##########################################"
cd CPN
./run.sh
cd ..

echo "##########################################"
echo "### Running SegLink                    ###"
echo "##########################################"
cd SegLink
./run.sh
cd ..

echo "##########################################"
echo "### Running CRNN                       ###"
echo "##########################################"
cd CRNN
./run.sh
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

