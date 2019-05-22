#!/bin/bash

cd ../DSSD

if [ ! -d "VOC0712" ]; then
    ./prepare_dataset.sh
fi

cd -
