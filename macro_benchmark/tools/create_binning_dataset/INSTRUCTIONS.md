The scripts here are used to create binning dataset for tensorflow and pytorch.

create_pretraining_data_tfrecord.py is used to create binning dataset for tensorflow.

create_pretraining_data_hdf5.py is used to create binning dataset for pytorch.

The input to both scritps are sharded data, wiki data or Chinese data.

For usage, please refer to run_create_pretraining_data.sh.

run_create_pretraining_data.sh runs only one thread which may be slow if you have a large dataset.

For large dataset, it is better to divide the dataset into a number of shards and run run_create_pretraining_data.sh for each shard. Use launch.sh to launch multiple processes to simultaneously process multiple shards. Be careful the number of processes should be limited to avoid out of memory issue.