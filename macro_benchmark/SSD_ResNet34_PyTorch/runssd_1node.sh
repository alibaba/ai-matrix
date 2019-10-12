#!/bin/bash

# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export WORLD_SIZE=2

python -m torch.distributed.launch --nproc_per_node=2 \
  train.py \
  --use-fp16 \
  --nhwc \
  --pad-input \
  --jit \
  --delay-allreduce \
  --opt-loss \
  --epochs 1 \
  --warmup-factor 0 \
  --no-save \
  --threshold=0.23 \
  --data coco \
  --evaluation 120000 160000 180000 200000 220000 240000 260000 280000

