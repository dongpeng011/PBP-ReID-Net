# encoding: utf-8
"""Training / testing loops of PBP-ReID.

One loop covers both regimes of the paper: a single dataset (``Table 2``) and a
union of datasets under a unified label space (``Table 3``).  There is no
distillation stage and no scene prompt -- PBP-ReID is trained end to end with a
cross-entropy loss and a triplet loss, exactly as described in Section 3.4.
"""

from .trainer import do_train, do_test

__all__ = ['do_train', 'do_test']
