# Changelog

All notable changes to this repository are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.0] — 2026-11-09

### Added
- `tests/` with 22 unit tests covering EASA, HPRA, GAL and the assembled model
  (band partition, per-band attention equivalence, fusion identity, ablations,
  checkpoint compatibility).
- GitHub Actions workflow (`ci.yml`): byte-compilation, unit tests, the CPU smoke
  test and the efficiency tools on every push.
- `docs/RESULTS.md` — every table of the paper with the command that produces it.
- `docs/FAQ.md` — installation, dataset paths, memory and reproducibility.
- `CONTRIBUTING.md`, `CHANGELOG.md`, a `Makefile` and issue templates.
- `requirements-ci.txt` for the CPU-only CI environment.

### Fixed
- Equation numbers in the docstrings and the README now follow the manuscript
  (`x̂` = 5, `t_i` = 6, `X_f` = 7, `x_g` = 8, `x_l` = 9, `F` = 10).
- `datasets/combine.py` no longer assumes that the scene ids are contiguous and
  start at 0.
- `utils/metrics.py` uses the keyword form of `addmm_`, removed in PyTorch 2.x.

## [1.0.0] — 2026-11-09

### Added
- Initial release: ViT-Base backbone with EASA (parameter-free body-prior
  anchoring), HPRA (hierarchical part-relation attention) and GAL (global–local
  additive fusion), single-scene and multi-scene training, evaluation on the
  Market-1501 / MSMT17 / Occ-Duke / PRCC / Celeb-ReID / SYSU-MM01 / MLR-CUHK03
  protocols, the smoke test and the efficiency tools.
