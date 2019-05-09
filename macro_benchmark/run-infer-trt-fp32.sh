#!/bin/bash

start=`date +%s%N`
start_date=`date`

echo "##########################################"
echo "### Running CNN_Caffe                  ###"
echo "##########################################"
cd CNN_Caffe
./run_test_tensorRT-5-docker-fp32.sh
cd ..

echo "##########################################"
echo "### Running CNN_Tensorflow             ###"
echo "##########################################"
cd CNN_Tensorflow
./infer-all-trt-synthetic-input-fp32.sh
cd ..

echo "##########################################"
echo "### Running SSD_Caffe                  ###"
echo "##########################################"
cd SSD_Caffe/tensorrt/sampleSSD
./infer-fp32.sh
cd ../../..


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
