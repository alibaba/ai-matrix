#!/bin/bash

current_path=`pwd`

sudo nvidia-docker exec -it aimatrix-tf bash -c "cd $current_path && ./prepare_dataset_tf.sh"
