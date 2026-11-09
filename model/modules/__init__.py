# encoding: utf-8
"""The three contributions of PBP-ReID.

* :class:`~model.modules.easa.BodyPriorAnchoring` -- parameter-free body prior
  (Sec. 3.1)
* :class:`~model.modules.hpra.HPRAttention` -- hierarchical part-relation
  attention (Sec. 3.2)
* :class:`~model.modules.gal.GlobalLocalAdditiveFusion` -- global-local
  additive fusion (Sec. 3.3)
"""

from model.modules.easa import BodyPriorAnchoring
from model.modules.gal import GlobalLocalAdditiveFusion
from model.modules.hpra import HPRAttention, PBPBlock

__all__ = [
    'BodyPriorAnchoring',
    'HPRAttention',
    'PBPBlock',
    'GlobalLocalAdditiveFusion',
]
