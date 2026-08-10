#!/bin/sh
# P1: Step reward/no dup penalty

set -eux

PYTHON=${PYTHON:-'python3 -u'}
export SCRIPT_DIR="$(cd $(dirname "$0"); pwd)"
TOP_DIR=$SCRIPT_DIR/../../
echo "TOP_DIR: $TOP_DIR"
CONFIG_YAML=$SCRIPT_DIR/config_SA_label3_1000_eqk_ale_eqexp_epi_0315.yaml ##
DATASET_DIR=$TOP_DIR/data
MODEL_DIR=$TOP_DIR/GNN_finetune_model
OUTDIR=$TOP_DIR/results/config_SA_label3_1000_eqk_ale_eqexp_epi_0315/ ##

mkdir -p $OUTDIR
export OUTDIR
export DATASET_DIR
export MODEL_DIR

START_TIME=$(date +%s)

$PYTHON $TOP_DIR/scripts/rl/train_ppo.py \
    yaml=$CONFIG_YAML \
    trainer.reward.score_until_dup_count=2 \
    trainer.reward.use_final_reward=false \
    > $OUTDIR/config_SA_label3_1000_eqk_ale_eqexp_epi_0315.log 2>&1 ##

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Script execution time: $ELAPSED_TIME seconds" >> $OUTDIR/config_SA_label3_1000_eqk_ale_eqexp_epi_0315.log ##