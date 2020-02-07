#!/bin/bash

export WORLD_SIZE=2

python -m torch.distributed.launch --nproc_per_node=2 \
  tools/train_mlperf.py \
  --config-file 'configs/e2e_mask_rcnn_R_50_FPN_1x.yaml' \
  DTYPE 'float16' \
  PATHS_CATALOG 'maskrcnn_benchmark/config/paths_catalog_dbcluster.py' \
  MODEL.WEIGHT '../coco/R-50.pkl' \
  DISABLE_REDUCED_LOGGING True \

