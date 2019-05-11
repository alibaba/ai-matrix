#!/bin/bash

if [ ! -d "dataset" ]; then
	mkdir dataset
fi

cd dataset
../nmt/scripts/wmt16_en_de.sh
cd ..

echo "Running md5 checksum on downloaded dataset ..."
if md5sum -c checksum.md5; then
        echo "Dataset checksum pass"
else
        echo "Dataset checksum fail"
        exit 1
fi

