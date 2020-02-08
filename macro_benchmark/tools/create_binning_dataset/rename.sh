#!/bin/bash

cd $1

for file in `ls`
do
	number=$(echo $file | tr -dc '0-9')
	printf -v number "%03d" $number
	filename="length_$number.tfrecord"
	echo $number
	echo $file
	echo $filename
	mv $file $filename
done

cd ..

