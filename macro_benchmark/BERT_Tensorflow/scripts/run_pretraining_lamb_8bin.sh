#!/usr/bin/env bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

start=`date +%s%N`
start_date=`date`

echo "Container nvidia build = " $NVIDIA_BUILD_ID

train_batch_size_phase1=${1:-270}
train_batch_size_phase2=${2:-8}
eval_batch_size=${3:-8}
learning_rate_phase1=${4:-"7.5e-4"}
learning_rate_phase2=${5:-"5e-4"}
precision=${6:-"fp16"}
use_xla=${7:-"true"}
num_gpus=${8:-1}
warmup_steps_phase1=${9:-"2000"}
warmup_steps_phase2=${10:-"200"}
train_steps=${11:-100000000000000}
save_checkpoints_steps=${12:-1000}
num_accumulation_steps_phase1=${13:-1}
num_accumulation_steps_phase2=${14:-1}
bert_model=${15:-"base"}
seq_len=${16:-16}

#DATA_DIR=data
#export DATA_DIR=$DATA_DIR

GBS1=$(expr $train_batch_size_phase1 \* $num_gpus \* $num_accumulation_steps_phase1)
GBS2=$(expr $train_batch_size_phase2 \* $num_gpus \* $num_accumulation_steps_phase2)
printf -v TAG "tf_bert_pretraining_lamb_%s_%s_gbs1%d_gbs2%d" "$bert_model" "$precision" $GBS1 $GBS2
DATESTAMP=`date +'%y%m%d%H%M%S'`

#Edit to save logs & checkpoints in a different directory
# RESULTS_DIR=${RESULTS_DIR:-results/${TAG}_${DATESTAMP}}
RESULTS_DIR=${RESULTS_DIR:-results/16}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"
export RESULTS_DIR=$RESULTS_DIR

input_files="$DATA_DIR/create_maxlen128_dupe2_MLM015_NSP_wikiSentSeg/wiki_dupe2_lowercase_8sec/16"
checkpoint="None"

printf -v SCRIPT_ARGS "%d %d %d %e %e %s %s %d %d %d %d %d %d %d %s %s" \
                      $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 \
                      $learning_rate_phase2 "$precision" "$use_xla" $num_gpus $warmup_steps_phase1 \
                      $warmup_steps_phase2 $train_steps $save_checkpoints_steps \
                      $num_accumulation_steps_phase1 $num_accumulation_steps_phase2 "$bert_model" $seq_len

# RUN PHASE 1
bash scripts/run_pretraining_lamb_phase1.sh $SCRIPT_ARGS $input_files $checkpoint |& tee -a $LOGFILE

end=`date +%s%N`
end_date=`date`
total_time=`bc <<< "scale = 0; ($end-$start)/1000000000"`
total_hours=`bc <<< "scale = 0; ${total_time}/3600"`
total_minutes=`bc <<< "sale = 0; (${total_time}%3600)/60"`
total_seconds=`bc <<< "scale = 0; ${total_time}%60"`
echo "Running started at ${start_date}"
echo "          ended at ${end_date}"
echo "Total running time is ${total_hours}h ${total_minutes}m ${total_seconds}s"

printf "\n\n\nFinished section 16\n\n\n"

prev_sec=16

########################################################################################################

for sec in 32 48 64 80 96 112 128; do

RESULTS_DIR="results/$sec"
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"
export RESULTS_DIR=$RESULTS_DIR

input_files="$DATA_DIR/create_maxlen128_dupe2_MLM015_NSP_wikiSentSeg/wiki_dupe2_lowercase_8sec/$sec"
checkpoint=`ls results/$prev_sec/phase_1/model.ckpt-*.index | sort -V | tail -1`
checkpoint=${checkpoint%'.index'}
seq_len=$sec

echo "$input_files"
echo "$checkpoint"
echo "$seq_len"

printf -v SCRIPT_ARGS "%d %d %d %e %e %s %s %d %d %d %d %d %d %d %s %s" \
                      $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 \
                      $learning_rate_phase2 "$precision" "$use_xla" $num_gpus $warmup_steps_phase1 \
                      $warmup_steps_phase2 $train_steps $save_checkpoints_steps \
                      $num_accumulation_steps_phase1 $num_accumulation_steps_phase2 "$bert_model" $seq_len

bash scripts/run_pretraining_lamb_phase1.sh $SCRIPT_ARGS $input_files $checkpoint |& tee -a $LOGFILE

prev_sec=$sec

end=`date +%s%N`
end_date=`date`
total_time=`bc <<< "scale = 0; ($end-$start)/1000000000"`
total_hours=`bc <<< "scale = 0; ${total_time}/3600"`
total_minutes=`bc <<< "sale = 0; (${total_time}%3600)/60"`
total_seconds=`bc <<< "scale = 0; ${total_time}%60"`
echo "Running started at ${start_date}"
echo "          ended at ${end_date}"
echo "Total running time is ${total_hours}h ${total_minutes}m ${total_seconds}s"

printf "\n\n\nFinished section $sec\n\n\n"

done

# RUN PHASE 2
#bash scripts/run_pretraining_lamb_phase2.sh $SCRIPT_ARGS |& tee -a $LOGFILE

end=`date +%s%N`
end_date=`date`
total_time=`bc <<< "scale = 0; ($end-$start)/1000000000"`
total_hours=`bc <<< "scale = 0; ${total_time}/3600"`
total_minutes=`bc <<< "sale = 0; (${total_time}%3600)/60"`
total_seconds=`bc <<< "scale = 0; ${total_time}%60"`
echo "Running started at ${start_date}"
echo "          ended at ${end_date}"
echo "Total running time is ${total_hours}h ${total_minutes}m ${total_seconds}s"

