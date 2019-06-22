#!/bin/bash

start=`date +%s%N`
start_date=`date`

echo "##########################################"
echo "### Running CNN_Tensorflow             ###"
echo "##########################################"
cd CNN_Tensorflow
models='alexnet googlenet vgg16 resnet50 resnet152 densenet121 synNet'
batchs='16 32 64'
is_trained=1
for md in $models
do
    for batch in $batchs
    do
    	if [[ ! -d log_${md}_${batch} ]]; then
    		is_trained=0
    	fi
    done
done
if [[ $is_trained == 0 ]]; then
	echo "Training has not been run for CNN_Tensorflow, run training now."
	rm -r log_*
	./train-all.sh
fi
./infer-all-synthetic-input.sh
cd ..

echo "##########################################"
echo "### Running DeepInterest               ###"
echo "##########################################"
cd DeepInterest
if [[ ! -d din/save_path ]]; then
	echo "Training has not been run for DeepInterest, run training now."
	./train.sh
fi
./infer.sh
cd ..

echo "##########################################"
echo "### Running Mask_RCNN                  ###"
echo "##########################################"
cd Mask_RCNN
./infer.sh
cd ..

echo "##########################################"
echo "### Running NMT                        ###"
echo "##########################################"
cd NMT
./infer.sh
cd ..

echo "##########################################"
echo "### Running SSD_VGG16_Caffe            ###"
echo "##########################################"
cd SSD_VGG16_Caffe
./infer.sh
cd ..

echo "##########################################"
echo "### Running SSD_ResNet18_Caffe         ###"
echo "##########################################"
cd SSD_ResNet18_Caffe
./infer.sh
cd ..

echo "##########################################"
echo "### Running DSSD                       ###"
echo "##########################################"
cd DSSD
./infer.sh
cd ..

echo "##########################################"
echo "### Running NCF                        ###"
echo "##########################################"
cd NCF
./infer.sh
cd ..

echo "##########################################"
echo "### Running DIEN                       ###"
echo "##########################################"
cd DIEN
./infer.sh
cd ..

echo "##########################################"
echo "### Running BERT_NVIDIA                ###"
echo "##########################################"
cd BERT_NVIDIA
./infer.sh
cd ..

echo "##########################################"
echo "### Running Faster_RCNN                ###"
echo "##########################################"
cd Faster_RCNN
./infer.sh
cd ..

echo "##########################################"
echo "### Running WideDeep                   ###"
echo "##########################################"
cd WideDeep
./infer.sh
cd ..

echo "##########################################"
echo "### Running CPN                        ###"
echo "##########################################"
cd CPN
./infer.sh
cd ..

echo "##########################################"
echo "### Running SegLink                    ###"
echo "##########################################"
cd SegLink
./infer.sh
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
