#!/bin/bash

nvidia-smi --query-gpu=gpu_name  --format=csv | tee -a system-info.txt
cat /proc/cpuinfo | grep 'model name' | uniq | tee -a system-info.txt
cat /proc/cpuinfo | grep 'processor' | wc -l | tee -a system-info.txt
cat /proc/meminfo | grep 'MemTotal'  | tee -a system-info.txt
