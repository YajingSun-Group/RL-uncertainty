#!/bin/sh

TOP_DIR="$(cd $(dirname "$0")/..; pwd)"
PYTHON=${PYTHON:-'python3 -u'}
DATASET_CSV=data/FORMAT_try.csv
OUTPUT_DIR=results/FORMAT_try
VOCAB=$OUTPUT_DIR/vocab.csv

mkdir -p $OUTPUT_DIR
$PYTHON $TOP_DIR/scripts/prep_dataset.py \
    --split 5 \
    -i $DATASET_CSV \
    --column smiles \
    -o $OUTPUT_DIR/train \
    -v $VOCAB
