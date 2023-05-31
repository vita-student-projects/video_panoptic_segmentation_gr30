#!/usr/bin/env bash

RESULTS_DIR=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/evaluate_vpq.py $RESULTS_DIR