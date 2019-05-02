#!/bin/bash

mkdir -p log_pressure
make allgemm
device_type=$1
if [ "$device_type" = "V100" ];then
    # Only run FP16 mul with FP16 accumulate on Tensorcore
    ./bin/allgemm FP16_TENSOR pressure | tee log_pressure/allgemm_FP16_TENSOR.csv
    # Only run FP16 mul with FP32 accumulate on Tensorcore
    ./bin/allgemm FP16_32_TENSOR pressure | tee log_pressure/allgemm_FP16_32_TENSOR.csv
    # Only run FP32 mul with FP32
    ./bin/allgemm FP32_CUDA pressure | tee log_pressure/allgemm_FP32_CUDA.csv
    # Only run FP16 mul with FP16
    ./bin/allgemm FP16_CUDA pressure | tee log_pressure/allgemm_FP16_CUDA.csv
fi

if [ "$device_type" = "P100" ];then
    # Only run FP32 mul with FP32
    ./bin/allgemm FP32_CUDA pressure | tee log_pressure/allgemm_FP32_CUDA.csv
    # Only run FP16 mul with FP16
    ./bin/allgemm FP16_CUDA pressure | tee log_pressure/allgemm_FP16_CUDA.csv
fi

if [ "$device_type" = "T4" ];then
    # Only run FP16 mul with FP16 accumulate on Tensorcore
    ./bin/allgemm FP16_TENSOR pressure | tee log_pressure/allgemm_FP16_TENSOR.csv
    # Only run FP16 mul with FP32 accumulate on Tensorcore
    ./bin/allgemm FP16_32_TENSOR pressure | tee log_pressure/allgemm_FP16_32_TENSOR.csv
    # Only run FP32 mul with FP32
    ./bin/allgemm FP32_CUDA pressure | tee log_pressure/allgemm_FP32_CUDA.csv
    # Only run FP16 mul with FP16
    ./bin/allgemm FP16_CUDA pressure | tee log_pressure/allgemm_FP16_CUDA.csv
    # Only run INT8 mul with INT32
    ./bin/igemm_imma pressure | tee log_pressure/igemm_int8.csv
fi
