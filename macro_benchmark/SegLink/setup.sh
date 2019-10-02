#!/bin/bash

if [ ! -f "/usr/local/lib/python3.6/dist-packages/tensorflow/libtensorflow_framework.so" ]; then
	ln -s /usr/local/lib/python3.6/dist-packages/tensorflow/libtensorflow_framework.so.1 /usr/local/lib/python3.6/dist-packages/tensorflow/libtensorflow_framework.so
fi

./manage.py build_op

if [ ! -d "model" ]; then
	mkdir model
fi


