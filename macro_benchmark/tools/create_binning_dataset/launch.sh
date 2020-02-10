#!/bin/bash

for i in {0..7}; do
	bash run_create_pretraining_data.sh $i &
done
