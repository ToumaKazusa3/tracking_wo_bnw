#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
GPUS_PER_NODE=$3
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u experiments/scripts/train_reid_mot15.py
