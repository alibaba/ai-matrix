#!/bin/bash

./nmt/scripts/download_iwslt15.sh dataset/en_vi

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi

