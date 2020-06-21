# please set data_dir by yourself
mkdir result
for i in {1..100}
do
    rm ./result/* -rf
    echo "Running $i times"
    mpiexec --allow-run-as-root --bind-to socket -np 2 python main.py --mode=train_and_evaluate --iter_unit=epoch --num_iter=50 --batch_size=256 --warmup_steps=100 --use_cosine_lr --label_smoothing 0.1 --lr_init=0.256 --lr_warmup_epochs=8 --momentum=0.875 --weight_decay=3.0517578125e-05 --use_tf_amp  --data_dir=/data/source_data/build_imagenet_data-rebuild/final_output/  --results_dir=./result | tee {1}.log
done
