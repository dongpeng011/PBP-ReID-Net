#!/usr/bin/env bash
# PBP-ReID -- single-scene regime (Table 2 of the paper).
#
#   bash bash/run_PBP-ReID.sh [GPU_ID] [DATASET_ROOT]
#
# Defaults: GPU 0, Market-1501 in data/market1501.
set -e

GPU_ID=${1:-0}
DATA_ROOT=${2:-data/market1501}
CONFIG=configs/PBP-ReID.yml
OUTPUT=logs/market1501

mkdir -p "${OUTPUT}"

python train.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "('${GPU_ID}')" \
    DATASETS.ROOTS "['${DATA_ROOT}']" \
    OUTPUT_DIR "${OUTPUT}"

# evaluate the last checkpoint (60 epochs by default)
python test.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "('${GPU_ID}')" \
    MODEL.PRETRAIN_CHOICE none \
    DATASETS.ROOTS "['${DATA_ROOT}']" \
    TEST.WEIGHT "${OUTPUT}/pbp_reid_60.pth" \
    OUTPUT_DIR "${OUTPUT}"
