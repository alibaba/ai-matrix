#!/bin/bash

./manage.py build_op

if [ ! -d "model" ]; then
	mkdir model
fi

cd tool/convert_caffe_model
./run.sh
cd -

