#!/bin/bash

make -j64
make py
make test -j64
make runtest -j64
