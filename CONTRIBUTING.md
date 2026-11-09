# Contributing to PBP-ReID

Thanks for your interest. This is a research repository that accompanies a
journal manuscript, so the priorities are **correctness** and **reproducibility**
rather than feature growth.

## Before opening an issue

1. Run `python tools/smoke_test.py`. Most installation problems surface there.
2. Check [`docs/FAQ.md`](docs/FAQ.md) for dataset paths, GPU memory and the
   known differences between the code and the paper.
3. Search the existing issues.

## Reporting a problem

Please use the issue template and include

* the command you ran (config file and command-line overrides),
* the full traceback,
* the output of
  `python -c "import torch, sys; print(sys.version); print(torch.__version__, torch.version.cuda)"`,
* whether `tools/smoke_test.py` passes.

## Pull requests

* Keep the change focused: one fix or one feature per pull request.
* Run the checks locally before pushing:

  ```bash
  python -m compileall -q config datasets loss model processor solver tools utils
  python -m pytest tests -q
  python tools/smoke_test.py
  ```

* New behaviour must come with a test under `tests/`. The existing tests check
  the equations of the paper (band partition, the two attention levels and the
  additive fusion), so keep them passing without weakening the assertions.
* Follow the existing style: 4-space indentation, 100-character lines, English
  comments, docstrings for public classes and functions, and no stray debugging
  output.
* Do not change the numerical results reported in the paper without saying so
  explicitly in the pull request.

## Scope

Bug fixes, documentation, additional dataset loaders, and support for new
backbones are welcome. Changes that alter the method itself should be discussed
in an issue first, because the repository is meant to reproduce the manuscript.
