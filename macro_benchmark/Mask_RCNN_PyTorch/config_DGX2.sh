#!/bin/bash

## DL params
EXTRA_PARAMS=""
EXTRA_CONFIG=(
               "SOLVER.BASE_LR"       "0.12"
               "SOLVER.MAX_ITER"      "42000"
               "SOLVER.WARMUP_FACTOR" "0.000192"
               "SOLVER.WARMUP_ITERS"  "625"
               "SOLVER.WARMUP_METHOD" "mlperf_linear"
               "SOLVER.STEPS"         "(12000, 16000)"
               "SOLVER.IMS_PER_BATCH" "96"
               "TEST.IMS_PER_BATCH" "16"
               "MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN" "6000"
               "NHWC" "True"   
             )

## System run parms
DGXNNODES=1
DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=04:00:00

## System config params
DGXNGPU=16
DGXSOCKETCORES=24
DGXNSOCKET=2
DGXHT=2         # HT is on is 2, HT off is 1
DGXIBDEVICES=''
BIND_LAUNCH=1
