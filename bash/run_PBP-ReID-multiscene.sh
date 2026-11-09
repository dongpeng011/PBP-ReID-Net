#!/usr/bin/env bash
# PBP-ReID -- multi-scene regime (Table 3 of the paper).
#
#   bash bash/run_PBP-ReID-multiscene.sh [GPU_ID]
#
# Edit the seven ROOTS below (they must be in the same order as COMBINE_NAMES in
# configs/PBP-ReID-multiscene.yml).
set -e

GPU_ID=${1:-0}
CONFIG=configs/PBP-ReID-multiscene.yml
OUTPUT=logs/multiscene

ROOTS="['data/MLR-CUHK03','data/market1501','data/MSMT17_V1','data/Celeb-reID','data/prcc','data/Occluded-DukeMTMC','data/SYSU-MM01']"

mkdir -p "${OUTPUT}"

python train.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "('${GPU_ID}')" \
    DATASETS.ROOTS "${ROOTS}" \
    OUTPUT_DIR "${OUTPUT}"

python test.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "('${GPU_ID}')" \
    MODEL.PRETRAIN_CHOICE none \
    DATASETS.ROOTS "${ROOTS}" \
    TEST.WEIGHT "${OUTPUT}/pbp_reid_60.pth" \
    OUTPUT_DIR "${OUTPUT}"
