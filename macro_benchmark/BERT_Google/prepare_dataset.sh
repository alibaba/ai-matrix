#!/bin/bash

if [ ! -d "glue_data" ]; then
	python download_glue_data.py
fi

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi

