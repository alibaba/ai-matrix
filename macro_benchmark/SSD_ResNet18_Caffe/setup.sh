#!/bin/bash

if [ ! -f /usr/local/lib/libprotobuf.so.9 ]; then
	cd ../DSSD
	./setup.sh
	cd -
fi
