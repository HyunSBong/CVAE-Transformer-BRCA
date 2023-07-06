#!/bin/bash

train_x="../data/tcga_brca_new_sampleid_train.csv"
test_x="../data/tcga_brca_new_sampleid_val.csv"

python attomics.py \
    --gpu_id 2 \
    --seed 42 \
    --train_x=$train_x \
    --val_x=$train_x
    