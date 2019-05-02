#!/bin/bash
md=$1
st=100
end=1000
num_batches=100
if [ ! -e test_max_bs ];then 
    mkdir test_max_bs
fi



# check result
checkResult(){
    res=$(grep -E "run out of memory|exception" $1 | wc -l)
    #echo $res 
    if [ $res -gt 0 ];then
    	echo "batch size = $2 does not work!"
	return 0
    else 
    	echo "batch size = $2 pass!"
        return 1
    fi
}
# binary search for max batch size
mbs=$st
while [ $(($end - $st))  -gt 32 ]; do
    batch=$(($end + $st))
    batch=$(($batch / 2))
    echo "st=$st end=$end batch=$batch"
    echo "----------------------------------------------------------------"
    echo "Running $md with batch size of $batch"
    echo "----------------------------------------------------------------"
    start=`date +%s%N`
    command="python ../nvcnn.py --model=$md \
                    --batch_size=$batch \
                    --num_gpus=1 \
                    --num_batches=$num_batches   \
                    --display_every=100  \
                    --log_dir=../../pretrained_models/aimatrix-pretrained-weights/CNN_Tensorflow/log_${md}_32-save \
                    --eval \
                    |& tee ./test_max_bs/result_${md}_${batch}.txt"
    echo $command
    eval $command
    checkResult "./test_max_bs/result_${md}_${batch}.txt" "$batch"
    if [ "$?" -eq 1 ];then
	st=$batch
        mbs=$st
    else	
 	end=$batch
    fi
done
echo "maximum batch size = $mbs"
