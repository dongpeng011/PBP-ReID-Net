# encoding: utf-8
"""GAL -- Global-Local Additive Fusion (paper, Sec. 3.3).

The identity descriptor is the *sum* of a global and a part-level term::

    x_g = GAP(X)                                   (9)
    x_l = Flatten(X_f) W_f + b_f                   (10)
    F   = x_g + x_l                                (11)

Because the two terms live in the same ``C``-dimensional space, the addition
needs no channel expansion and no fully connected layer after concatenation,
which is the difference with respect to PCB-style fusion.  The only parameters
are ``W_f`` and ``b_f``, i.e. ``P * C^2 + C``.
"""

import torch
import torch.nn as nn

from model.vit_utils import trunc_normal_


class GlobalLocalAdditiveFusion(nn.Module):
    """Additive fusion of a global descriptor and ``P`` band descriptors.

    Parameters
    ----------
    dim : int
        Channel dimension ``C`` of the backbone.
    part_num : int
        Number of bands ``P``.
    bias : bool
        Whether to add ``b_f`` (Eq. 10).
    """

    def __init__(self, dim, part_num=3, bias=True):
        super(GlobalLocalAdditiveFusion, self).__init__()
        self.dim = dim
        self.part_num = part_num
        self.proj = nn.Linear(dim * part_num, dim, bias=bias)
        trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x_global, x_part):
        """Fuse ``x_global`` ``[B, C]`` with ``x_part`` ``[B, P, C]``.

        ``x_part`` is flattened in band order, so ``W_f`` of Eq. (10) is exactly
        the weight matrix of :attr:`proj`.
        """
        if x_part.dim() == 2:
            x_part = x_part.unsqueeze(1)
        if x_part.shape[1] != self.part_num:
            raise ValueError('Expected {} bands, got {}'.format(self.part_num, x_part.shape[1]))
        x_local = self.proj(x_part.reshape(x_part.shape[0], -1))
        return x_global + x_local

    def extra_repr(self):
        return 'dim={}, part_num={}'.format(self.dim, self.part_num)
