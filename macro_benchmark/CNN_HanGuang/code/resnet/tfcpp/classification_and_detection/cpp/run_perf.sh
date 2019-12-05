#!/bin/bash
valid=0
if [ "$1" = "SingleStream" ]; then
  valid=1
elif [ "$1" = "MultiStream" ]; then
  valid=1
elif [ "$1" = "Server" ]; then
  valid=1
elif [ "$1" = "Offline" ]; then
  valid=1
fi

if [ "$valid" = 0 ]; then
  echo "Invalid scenario" $1
  exit 1
fi

echo "Run performance test on scenario" $1

scenario=$1
shift

export GLOG_v=0
./classification --model=${MODEL_DIR}/npu.pb --scenario=${scenario} --mlperf-config=../../../../../measurements/alibaba_HanGuang/resnet/${scenario}/mlperf.conf --user-config=../../../../../measurements/alibaba_HanGuang/resnet/${scenario}/user.conf $*
