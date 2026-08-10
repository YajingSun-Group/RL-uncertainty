#!/bin/sh
set -eux
TOP_DIR="$(cd $(dirname "$0")/../..; pwd)"
PYTHON=${PYTHON:-'python3 -u'}
OUTDIR=results/FORMAT
DATASET_DIR=results/FORMAT
DATASET=$DATASET_DIR/worker
VOCAB=$DATASET_DIR/vocab.csv

mkdir -p $OUTDIR

$PYTHON $TOP_DIR/scripts/rl/pretrain_policy.py \
    -v $VOCAB \
    --input_pkl $DATASET \
    --epoch 10 \
    --snap_freq 100 \
    --snap_name "final_snap.pt" \
    --resume "$OUTDIR/final_snap.pt" \
    --save_best_snapshot \
    --log_interval 1000 \
    --log_interval_unit iteration \
    --batch_size 128 \
    --hidden_size 128 \
    --num_workers=2 \
    --n_total=300000 \
    -g 0 \
    -o $OUTDIR \
