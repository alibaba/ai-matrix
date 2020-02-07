#!/bin/bash

## DL params
EXTRA_PARAMS=""
EXTRA_CONFIG=(
               "SOLVER.BASE_LR"       "0.24"
               "SOLVER.MAX_ITER"      "40000"
               "SOLVER.WARMUP_FACTOR" "0.000192"
               "SOLVER.WARMUP_ITERS"  "1250"
               "SOLVER.WARMUP_METHOD" "mlperf_linear"
               "SOLVER.STEPS"         "(7000, 9333)"
               "SOLVER.IMS_PER_BATCH"  "192"
               "TEST.IMS_PER_BATCH" "192"
               "MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN" "1000"
               "NHWC" "True"
             )

## System run parms
DGXNNODES=12
DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=08:00:00

## System config params
DGXNGPU=16
DGXSOCKETCORES=24
DGXNSOCKET=2
DGXHT=2 	# HT is on is 2, HT off is 1
DGXIBDEVICES='--device=/dev/infiniband/rdma_cm --device=/dev/infiniband/ucm10 --device=/dev/infiniband/ucm9 --device=/dev/infiniband/ucm8 --device=/dev/infiniband/ucm7 --device=/dev/infiniband/ucm4 --device=/dev/infiniband/ucm3 --device=/dev/infiniband/ucm2 --device=/dev/infiniband/ucm1 --device=/dev/infiniband/uverbs10 --device=/dev/infiniband/uverbs9 --device=/dev/infiniband/uverbs8 --device=/dev/infiniband/uverbs7 --device=/dev/infiniband/uverbs4 --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/issm10 --device=/dev/infiniband/umad10 --device=/dev/infiniband/issm9 --device=/dev/infiniband/umad9 --device=/dev/infiniband/issm8 --device=/dev/infiniband/umad8 --device=/dev/infiniband/issm7 --device=/dev/infiniband/umad7 --device=/dev/infiniband/issm4 --device=/dev/infiniband/umad4 --device=/dev/infiniband/issm3 --device=/dev/infiniband/umad3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/umad2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/umad1'
BIND_LAUNCH=1
