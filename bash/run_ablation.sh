#!/usr/bin/env bash
# The four configurations of the ablation table (Table 5).
#
#   bash bash/run_ablation.sh [GPU_ID] [DATASET_ROOT]
#
# Each run writes to its own log directory.  Adjust SOLVER.MAX_EPOCHS to the
# budget you want to spend; the default of the config file is 60.
set -e

GPU_ID=${1:-0}
DATA_ROOT=${2:-data/market1501}
CONFIG=configs/PBP-ReID.yml

run_variant () {
    local name=$1
    shift
    echo "=== ${name} ==="
    python train.py --config_file "${CONFIG}" \
        MODEL.DEVICE_ID "('${GPU_ID}')" \
        DATASETS.ROOTS "['${DATA_ROOT}']" \
        OUTPUT_DIR "logs/ablation/${name}" \
        "$@"
}

# (a) ViT-Base reference: full-token attention everywhere, no additive fusion
run_variant reference MODEL.HPRA_START 12 MODEL.GAL False

# (b) HPRA without GAL
run_variant hpra_only MODEL.GAL False

# (c) GAL without HPRA (bands pooled directly, no part-wise attention)
run_variant gal_only MODEL.HPRA_START 12

# (d) full model
run_variant full
