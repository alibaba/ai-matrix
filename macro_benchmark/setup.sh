#!/bin/bash

current_path=`pwd`

sudo nvidia-docker exec -it aimatrix-tf bash -c "cd $current_path && ./setup_tf.sh"
