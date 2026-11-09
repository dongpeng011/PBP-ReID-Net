# FAQ

## Installation

**`ImportError: No module named yacs`**
`yacs` is optional. `config/cfg_node.py` provides the same API and is used
automatically when yacs is absent. Installing yacs (`pip install yacs`) also
works and takes precedence.

**`ImportError: cannot import name 'RandomErasing' from 'timm.data'`**
Install `timm>=0.9` (`pip install timm==0.9.16`). It is only used by the
training transform.

**`RuntimeError: CUDA out of memory`**
The model has 88 M parameters and trains at 256 × 128 with batch 64. Reduce
`SOLVER.IMS_PER_BATCH` (e.g. to 32) and, if needed, `DATALOADER.NUM_INSTANCE`.
Mixed precision is enabled automatically on CUDA.

**`torch.cuda.is_available()` is False**
The code runs on CPU for the smoke test, but training on CPU is not practical.
Install a CUDA build of PyTorch that matches your driver.

## Data

**What layout does each loader expect?**
See the table in `README.md` and the docstring of each file in `datasets/`.
Market-1501 style folders (`bounding_box_train/`, `query/`, `bounding_box_test/`)
are the simplest starting point; `tools/make_dummy_dataset.py` generates one.

**`RuntimeError: 'data/xxx/train' is not available`**
`DATASETS.ROOTS` still points at the placeholder path. Override it on the
command line, e.g.
`DATASETS.ROOTS "['/data/market1501']"`, or edit the config file.

**Can I train on a single dataset without the `combine` machinery?**
Yes — set `DATASETS.COMBINE_NAMES` to one entry and `COMBINE_TYPE` to one scene
id. That is exactly what `configs/PBP-ReID.yml` does.

## Model

**How do I get the plain ViT-Base reference row of Table 4?**
`MODEL.HPRA_START 12 MODEL.GAL False`. With `HPRA_START = 12` no block uses HPRA;
with `GAL False` the descriptor is the global average pooling of the tokens.

**What does `MODEL.HPRA_START` mean?**
It is the index of the first block whose full-token attention is replaced by
HPRA. `6` (the default) uses HPRA in blocks 6–11, `0` in every block, `12` in
none.

**Why is `MODEL.HPRA_REL_SHARE` `True` by default?**
With separate level-2 projections the model grows by 14.2 M parameters; the
parameter count reported in the paper corresponds to sharing them. Set it to
`False` for the literal reading of Eq. (7).

**Can I load the official ViT-Base checkpoint?**
Yes. `MODEL.PRETRAIN_PATH` accepts ImageNet or self-supervised ViT-Base weights,
also wrapped in `{'model': ...}` / `{'state_dict': ...}` and with `module.` or
`base.` prefixes. The positional embedding is interpolated to the 16 × 8 grid
and the layers that are new in PBP-ReID are initialised from scratch
(`model/pbp_reid.py:load_param`).

## Results

**The parameter count I get is 88.1 M, the paper says 87.3 M.**
The difference is the shape of `W_f`: the code follows Eq. (9) with
`W_f ∈ ℝ^{PC×C}` (1.77 M), the paper's number corresponds to a `C×C` matrix
shared across bands. See the reproducibility notes in `README.md`.

**Can I reproduce the accuracy tables?**
The tables are the numbers of the manuscript; the repository does not ship
trained weights, so reproducing them requires training on the benchmarks with
the same pre-trained backbone.

**Why is the model not faster than ViT-Base on my GPU?**
At 256 × 128 the sequence has 128 tokens and attention is only a small fraction
of the FLOPs of a ViT-Base block, so replacing two thirds of it saves about 1 %
of the total compute. The gain of HPRA is in the attention-score work
(`N² → N²/P`), not in wall-clock latency. See Table 1 of the paper.

## Contributing

Please read [`CONTRIBUTING.md`](../CONTRIBUTING.md); run
`python -m pytest tests -q` and `python tools/smoke_test.py` before opening a
pull request.
