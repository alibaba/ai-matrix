#!/bin/bash

start=`date +%s%N`
start_date=`date`

echo "##########################################"
echo "### Running SSD_ResNet34_PyTorch       ###"
echo "##########################################"
cd SSD_ResNet34_PyTorch
./run.sh
cd ..

echo "##########################################"
echo "### Running BERT_PyTorch               ###"
echo "##########################################"
cd BERT_PyTorch
./run.sh
cd ..

end=`date +%s%N`
end_date=`date`
total_time=`bc <<< "scale = 0; ($end-$start)/1000000000"`
total_hours=`bc <<< "scale = 0; ${total_time}/3600"`
total_minutes=`bc <<< "sale = 0; (${total_time}%3600)/60"`
total_seconds=`bc <<< "scale = 0; ${total_time}%60"`
echo "Running started at ${start_date}"
echo "          ended at ${end_date}"
echo "Total running time is ${total_hours}h ${total_minutes}m ${total_seconds}s"
