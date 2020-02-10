#!/usr/bin/env bash

# Download SQUAD
cd data/squad && . squad_download.sh
cd ../..

# Download pretrained_models
cd data/pretrained_models_google && python3 download_models.py
cd ../..

# WIKI Download, set config in data_generators/wikipedia_corpus/config.sh
#cd /workspace/bert/data/wikipedia_corpus && . run_preprocessing.sh

#cd /workspace/bert/data/bookcorpus && . run_preprocessing.sh

