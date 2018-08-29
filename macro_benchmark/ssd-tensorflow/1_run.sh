#!/bin/bash
mkdir results
python ssd_notebook.py | tee ./results/result.txt
