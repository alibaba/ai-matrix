#!/bin/bash
mkdir results
python demo.py --batch_size=4 | tee ./results/result_4.txt
python demo.py --batch_size=8 | tee ./results/result_8.txt
python demo.py --batch_size=16 | tee ./results/result_16.txt


