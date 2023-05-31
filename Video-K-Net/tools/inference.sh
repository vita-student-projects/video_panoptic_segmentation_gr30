#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
RESULTS_DIR=$3
CONDA_ENV=${4:-base}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
/workspace/miniconda3/bin/conda run -n $CONDA_ENV --no-capture-output python $(dirname "$0")/inference.py $CONFIG $CHECKPOINT --show-dir $RESULTS_DIR ${@:5}