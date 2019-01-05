#!/bin/bash

mkdir -p log
python cnn_ops.py -op_name conv -times 10 -interval 1
sleep 1
python cnn_ops.py -op_name fc -times 10 -interval 1
sleep 1
python cnn_ops.py -op_name max_pool -times 10 -interval 1
sleep 1
python cnn_ops.py -op_name avg_pool -times 10 -interval 1
sleep 1
python cnn_ops.py -op_name bn -times 10 -interval 1
sleep 1
# python cnn_ops.py -op_name lrn -times 10 -interval 1
sleep 1
python cnn_ops.py -op_name relu -times 100 -interval 1 
sleep 1
python cnn_ops.py -op_name sigmoid -times 100 -interval 1 
sleep 1
python cnn_ops.py -op_name softmax -times 100 -interval 1
sleep 1
python cnn_ops.py -op_name tanh -times 100 -interval 1 


