#!/bin/bash

CURRENT_DIR=`dirname $0`
. $CURRENT_DIR/../../scripts/set_caffe_module_env.sh
ds_name="float16"
thread_num="8"

do_run()
{
    echo "----------------------"
    echo "multiple core"
    echo "using prototxt: $proto_file"
    echo "using model:    $model_file"
    echo "dataparallel:  $dp,  modelparallel:  $mp,  threadnum:  ${thread_num}"

    #first remove any offline model
    /bin/rm offline.cambricon* &> /dev/null

    log_file=$(echo $proto_file | sed 's/prototxt$/log/' | sed 's/^.*\///')
    echo > $CURRENT_DIR/$log_file

    genoff_cmd="$CURRENT_DIR/../../build/tools/caffe${SUFFIX} genoff -model $proto_file -weights $model_file -mcore MLU100 -model_parallel $mp"

    run_cmd="$CURRENT_DIR/clas_offline_multicore$SUFFIX -offlinemodel $CURRENT_DIR/offline.cambricon -images $FILE_LIST -labels $CURRENT_DIR/synset_words.txt -threads ${thread_num} -dataparallel $dp"

    echo "genoff_cmd: $genoff_cmd" &>> $CURRENT_DIR/$log_file
    echo "run_cmd: $run_cmd" &>> $CURRENT_DIR/$log_file

    echo "generating offline model..."
    eval "$genoff_cmd"

    if [[ "$?" -eq 0 ]]; then
        echo -e "running multicore offline test...\n"
        eval "$run_cmd"
        #tail -n 5 $CURRENT_DIR/$log_file
        grep "Global accuracy : $" -A 4 $CURRENT_DIR/$log_file
    else
        echo "generating offline model failed!"
    fi
}

desp_list=(
    dense
)

batch_list=(
    1batch
)
dpmp_list=(
    '4 1'
)
network_list=(
    densenet121
    googlenet-bvlc
)


for network in "${network_list[@]}"; do
    for desp in "${desp_list[@]}"; do
        model_file=$CAFFE_MODULES_DIR/ai-matrix/${network}_${ds_name}_${desp}.caffemodel
        echo -e "\n===================================================="
        echo "running ${network} offline - ${ds_name},${desp}..."

        for batch in "${batch_list[@]}"; do
            for proto_file in $CAFFE_MODULES_DIR/ai-matrix/${network}_${ds_name}*${desp}_${batch}.prototxt
            do
                for dpmp in "${dpmp_list[@]}"; do
                    dp=${dpmp:0:1}
                    mp=${dpmp:2:1}
                    do_run
                done
            done
        done
    done
done
