#!/bin/bash
models=${1:-resnet50}  
batchs='16 32 64'

function usage {
    echo "Usage: $0 [-m models] [-b batch_size]"
    echo "  -m  models to run. By default, $models"
    echo "  -b  batch sizes to sweep (should be >=1). By default, sweep $batchs"
}

while getopts "m:b:h" opt; do
    case ${opt} in
        m )
            models=$OPTARG
            ;;
        b )
            batchs=$OPTARG
            ;;
        h )
            usage
            exit
            ;;
        \? )
            echo "Invalid option. See the usage below" 1>&2
            usage
            ;;
    esac
done
shift $((OPTIND-1))

models=${models//[:,;-]/" "}
batchs=${batchs//[:,;-]/" "}
echo "models=$models"
echo "batch_size=$batchs"

for batch in $batchs
do
    python nvcnn.py --model=$models \
                --batch_size=$batch \
                --num_gpus=1 \
                --num_batches=500   \
                --display_every=100  \
                --log_dir=./log_${models}_${batch}  \
                --eval   
done
