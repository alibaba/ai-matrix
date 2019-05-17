#!/bin/bash

echo "################################################"
echo "Train the SSD model"
echo "################################################"

python examples/ssd/ssd_pascal_resnet_321.py --batch_size=$1 --num_accelerators=$2 --max_iter=$3

echo "################################################"
echo "Train the DSSD model"
echo "################################################"

python examples/ssd/ssd_pascal_resnet_deconv_321.py --batch_size=$1 --num_accelerators=$2 --max_iter=$3

echo "################################################"
echo "Fine tune the entire model"
echo "################################################"

python examples/ssd/ssd_pascal_resnet_deconv_ft_321.py --batch_size=$1 --num_accelerators=$2 --max_iter=$3
