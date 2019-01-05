#!/bin/bash
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

DATA_LOCATION=/data1/AIMatrixDataset/CommonVoice
CHECKPOINT_DIR=checkpoints

python -u DeepSpeech.py \
  --train_files "$DATA_LOCATION/cv-valid-train.csv" \
  --dev_files "$DATA_LOCATION/cv-valid-dev.csv" \
  --test_files "$DATA_LOCATION/cv-valid-test.csv" \
  --train_batch_size 32 \
  --dev_batch_size 32 \
  --test_batch_size 32 \
  --epoch 1 \
  --steps_p_epoch=1000 \
  --checkpoint_dir "$CHECKPOINT_DIR"
