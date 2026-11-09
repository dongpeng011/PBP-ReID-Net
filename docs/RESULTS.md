# Results of the paper

This page lists every table of the manuscript together with the command that
produces it, so that a reader can check what is and what is not reproducible
with the released code. **Trained weights are not distributed**, therefore the
accuracy tables are the numbers reported in the paper, while the efficiency
tables are measured with this repository.

## Table 2 — single-scene regime

mAP / Rank-1 (%), as reported in the manuscript.

| Method | Market-1501 | MSMT17 | Occ-Duke | SYSU-MM01 | MLR-CUHK03 |
|---|:---:|:---:|:---:|:---:|:---:|
| TransReID | 81.0 / 92.5 | 35.8 / 57.8 | 60.0 / 73.5 | 36.9 / 38.6 | 85.5 / 90.3 |
| VersReID | 82.0 / 92.9 | 37.1 / 59.9 | 60.9 / 74.4 | 37.8 / 39.5 | 85.7 / 90.9 |
| **PBP-ReID** | **82.7 / 93.4** | **41.4 / 66.0** | **62.1 / 75.6** | **39.4 / 40.7** | **87.7 / 91.9** |

Reproduce one cell:

```bash
python train.py --config_file configs/PBP-ReID.yml \
    DATASETS.ROOTS "['data/market1501']" OUTPUT_DIR logs/market1501
python test.py --config_file configs/PBP-ReID.yml \
    DATASETS.ROOTS "['data/market1501']" MODEL.PRETRAIN_CHOICE none \
    TEST.WEIGHT logs/market1501/pbp_reid_60.pth OUTPUT_DIR logs/market1501
```

## Table 3 — multi-scene regime

| Method | Market-1501 | MSMT17 | Celeb-ReID | PRCC |
|---|:---:|:---:|:---:|:---:|
| TransReID | 78.5 / 91.8 | 38.2 / 61.8 | 7.0 / 47.1 | 36.9 / 51.4 |
| VersReID | 79.6 / 92.2 | 39.5 / 62.9 | 7.8 / 48.6 | 38.3 / 55.8 |
| **PBP-ReID** | **81.0 / 92.6** | **42.3 / 66.4** | **8.3 / 49.4** | **40.3 / 53.1** |

```bash
python train.py --config_file configs/PBP-ReID-multiscene.yml \
    MODEL.DEVICE_ID "('0')" OUTPUT_DIR logs/multiscene
python test.py --config_file configs/PBP-ReID-multiscene.yml \
    MODEL.PRETRAIN_CHOICE none TEST.WEIGHT logs/multiscene/pbp_reid_60.pth \
    OUTPUT_DIR logs/multiscene
```

## Table 4 — computational cost

Measured with this repository on one **RTX 3090**, 256 × 128 crop, batch 64:

| Model | Params (M) | FLOPs (G) | Latency (ms) | FPS |
|---|:---:|:---:|:---:|:---:|
| ViT-B reference | 86.32 | 22.68 | 1.52 | 659 |
| **PBP-ReID** | **88.10** | **22.46** | **1.55** | **647** |

```bash
python tools/measure_speed.py --batch-size 64 --iterations 100
python tools/count_params_flops.py
```

See `docs/FAQ.md` and the "Reproducibility notes" section of the README for the
three places where these values differ from the numbers printed in the paper
(parameter count, level-2 projections, FLOPs ratio).

## Table 5 — ablation

| Configuration | Flag |
|---|---|
| ViT-B reference | `MODEL.HPRA_START 12 MODEL.GAL False` |
| + HPRA only | `MODEL.GAL False` |
| + GAL only | `MODEL.HPRA_START 12` |
| Full model | *(default)* |

`bash/run_ablation.sh` runs the four configurations in sequence; each one writes
to its own directory under `logs/ablation/`.

## Table 6 — leave-one-out contribution

Computed from Table 5 as the difference between the full model and the
configuration without one stage; no separate run is required.

## Table 7 — statistical reliability

Three independent runs per configuration, paired *t*-test against VersReID. The
numbers are the ones reported in the paper; to regenerate them, train each
configuration three times with `SOLVER.SEED 1234`, `1235`, `1236` and read the
mAP values from `logs/*/train_log.txt`:

```bash
for seed in 1234 1235 1236; do
  python train.py --config_file configs/PBP-ReID.yml \
      SOLVER.SEED $seed OUTPUT_DIR logs/seed_$seed
done
```

## Table 8 — deployment case study

Throughput and per-frame latency of the YOLOv12x + ReID + tracking pipelines on
three concurrent 1080p streams, measured on the authors' deployment machine.
It is a platform measurement, not something the repository can re-run.

## Figures 11 and 12 — speed and latency

Both figures plot the values of Table 4 (throughput versus mAP, and the latency
breakdown per stage). They can be regenerated with matplotlib from the CSV you
obtain by running `tools/measure_speed.py` on the same machine.
