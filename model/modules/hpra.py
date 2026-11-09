# encoding: utf-8
"""HPRA -- Hierarchical Part-Relation Attention (paper, Sec. 3.2).

Two levels of attention are applied to the tokens of the body bands produced by
EASA (``model/modules/easa.py``):

* **level 1 (intra-band).**  Self-attention is applied to every band
  independently with a *single set of projections shared by all bands*
  (``W_Q, W_K, W_V, W_O``).  Because the bands are disjoint and contain
  ``N/P`` tokens each, the number of attention-score entries drops from
  ``N^2`` to ``P * (N/P)^2 = N^2 / P`` -- a 66 % reduction for ``P = 3``
  (Table 1 of the paper).  Sharing the projections is what keeps the parameter
  count of this level identical to one plain attention layer.
* **level 2 (inter-band).**  Each refined band is pooled to a single token
  ``t_i``; a second, tiny attention layer reasons over the ``P`` band
  descriptors and returns ``X_f``.  ``X_f`` is the part-level representation
  consumed by GAL (Eq. 9) and its cost is negligible because ``P = 3``.

The module is a drop-in replacement for :class:`model.vit_utils.Attention`: it
keeps the ``qkv`` / ``proj`` attribute names so that a public ViT-Base
checkpoint can still be loaded.
"""

import torch
import torch.nn as nn

from model.vit_utils import Attention, DropPath, Mlp, multi_head_attention, trunc_normal_


class HPRAttention(nn.Module):
    """Hierarchical part-relation attention.

    Parameters
    ----------
    dim : int
        Embedding dimension ``C``.
    num_heads : int
        Heads of the intra-band attention (level 1).
    part_num : int
        Number of bands ``P``.
    band_indices : list[torch.LongTensor]
        Token indices of every band, taken from :class:`BodyPriorAnchoring`.
    rel_heads : int
        Heads of the inter-band attention (level 2).
    rel_share : bool
        If ``True`` the inter-band attention reuses the projections of level 1
        instead of adding a second set.  Default ``False``, i.e. the separate
        ``W_Q', W_K', W_V'`` of Eq. (7).
    rel_norm : bool
        Layer-normalise the band descriptors before level 2.  This only
        stabilises training and adds ``2C`` parameters.
    num_prefix : int
        Number of leading tokens that are not part of the body bands (the class
        token).  They bypass both attention levels and are concatenated back.
    """

    def __init__(self, dim, num_heads=12, part_num=3, band_indices=None,
                 rel_heads=4, rel_share=False, rel_norm=True, num_prefix=1,
                 qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super(HPRAttention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError('dim {} must be divisible by num_heads {}'.format(dim, num_heads))
        if dim % rel_heads != 0:
            raise ValueError('dim {} must be divisible by rel_heads {}'.format(dim, rel_heads))

        self.dim = dim
        self.num_prefix = int(num_prefix)
        self.num_heads = num_heads
        self.part_num = part_num
        self.rel_share = rel_share
        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop

        # ---- level 1: shared projections over all bands
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # ---- level 2: compact attention over the P band descriptors
        if rel_share:
            self.qkv_rel = None
            self.proj_rel = None
        else:
            self.qkv_rel = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj_rel = nn.Linear(dim, dim)
        self.rel_num_heads = rel_heads
        self.rel_scale = (dim // rel_heads) ** -0.5
        self.norm_rel = nn.LayerNorm(dim, eps=1e-6) if rel_norm else nn.Identity()
        self.rel_attn_drop = nn.Dropout(attn_drop)
        self.rel_proj_drop = nn.Dropout(proj_drop)

        if band_indices is None:
            raise ValueError('band_indices must be provided by the anchoring module')
        for band, index in enumerate(band_indices):
            self.register_buffer('band_index_{}'.format(band), index, persistent=False)
        self.num_bands = len(band_indices)
        self.band_sizes = [int(index.numel()) for index in band_indices]

        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self):
        trunc_normal_(self.qkv.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        trunc_normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, 0)
        if self.qkv_rel is not None:
            trunc_normal_(self.qkv_rel.weight, std=0.02)
            nn.init.constant_(self.qkv_rel.bias, 0)
            trunc_normal_(self.proj_rel.weight, std=0.02)
            nn.init.constant_(self.proj_rel.bias, 0)

    # ------------------------------------------------------------------ API
    def band_indices(self, device):
        return [getattr(self, 'band_index_{}'.format(band)).to(device)
                for band in range(self.num_bands)]

    def _split(self, tokens):
        # every band is a contiguous slice of the row-major token order, hence
        # index_select followed by cat are exact inverses of each other
        return [tokens.index_select(1, index) for index in self.band_indices(tokens.device)]

    @staticmethod
    def _mhsa(query, key, value, num_heads, scale, dropout_p):
        # shared kernel with the standard attention block (see model.vit_utils)
        out = multi_head_attention(query, key, value, num_heads, scale, dropout_p)
        return out.transpose(1, 2).reshape(query.shape[0], query.shape[1], query.shape[2])

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : Tensor ``[B, 1 + N, C]``
            Token sequence *after* the pre-attention LayerNorm of the block.
            The first ``num_prefix`` tokens (the class token) are not part of
            the body bands.

        Returns
        -------
        x_out : Tensor ``[B, 1 + N, C]``
            Tokens updated by the intra-band attention (to be added to the
            residual stream by the caller).
        x_f : Tensor ``[B, P, C]``
            Band-relation representation ``X_f`` used by GAL.
        """
        prefix = x[:, :self.num_prefix]
        body = x[:, self.num_prefix:]

        # ---------------------------------------------------------- level 1
        # The projections are shared by all bands, so applying them to the full
        # sequence and then slicing the bands is *identical* to computing them
        # band by band (a linear layer acts on every token independently), but
        # it issues a single large GEMM instead of P small ones.
        qkv = self.qkv(body)
        query, key, value = qkv.chunk(3, dim=-1)
        # EASA produces contiguous bands, so torch.split yields views without
        # any copy; this is what keeps the per-band attention cheap.
        query_bands = torch.split(query, self.band_sizes, dim=1)
        key_bands = torch.split(key, self.band_sizes, dim=1)
        value_bands = torch.split(value, self.band_sizes, dim=1)
        token_bands = torch.split(body, self.band_sizes, dim=1)

        refined_bands = []
        for band_query, band_key, band_value, band_tokens in zip(
                query_bands, key_bands, value_bands, token_bands):
            attended = self._mhsa(band_query, band_key, band_value, self.num_heads,
                                  self.scale, self.attn_drop.p if self.training else 0.0)
            refined_bands.append(attended + band_tokens)                       # Eq. (5)

        x_body = torch.cat(refined_bands, dim=1)
        x_body = self.proj(x_body)
        x_body = self.proj_drop(x_body)
        x_out = torch.cat([prefix, x_body], dim=1)                # inverse of EASA.split

        # ---------------------------------------------------------- level 2
        # t_i = GAP(X_hat_i)                                       Eq. (6)
        descriptors = torch.stack([band.mean(dim=1) for band in refined_bands], dim=1)
        band_tokens = self.norm_rel(descriptors)

        if self.rel_share:
            qkv = self.qkv(band_tokens)
            query, key, value = qkv.chunk(3, dim=-1)
            attended = self._mhsa(query, key, value, self.num_heads,
                                  self.scale, self.rel_attn_drop.p if self.training else 0.0)
            attended = self.proj(attended)
        else:
            qkv = self.qkv_rel(band_tokens)
            query, key, value = qkv.chunk(3, dim=-1)
            attended = self._mhsa(query, key, value, self.rel_num_heads,
                                  self.rel_scale, self.rel_attn_drop.p if self.training else 0.0)
            attended = self.proj_rel(attended)

        x_f = descriptors + self.rel_proj_drop(attended)           # Eq. (7)
        return x_out, x_f

    def extra_repr(self):
        return ('dim={}, num_heads={}, part_num={}, rel_heads={}, rel_share={}, '
                'num_prefix={}'.format(self.dim, self.num_heads, self.part_num,
                                       self.rel_num_heads, self.rel_share,
                                       self.num_prefix))


class PBPBlock(nn.Module):
    """A transformer block whose attention is either standard or HPRA.

    ``norm1 -> attention -> residual`` followed by ``norm2 -> MLP -> residual``,
    identical to a ViT block apart from the attention operator.  HPRA blocks
    additionally return ``X_f`` so that the backbone can hand the last one to
    GAL.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, hpra=False, part_num=3, band_indices=None,
                 rel_heads=4, rel_share=False, rel_norm=True, num_prefix=1):
        super(PBPBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.hpra = hpra
        if hpra:
            self.attn = HPRAttention(
                dim=dim, num_heads=num_heads, part_num=part_num,
                band_indices=band_indices, rel_heads=rel_heads,
                rel_share=rel_share, rel_norm=rel_norm, num_prefix=num_prefix,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        """Return ``(tokens, x_f)``; ``x_f`` is ``None`` for standard blocks."""
        if self.hpra:
            attended, x_f = self.attn(self.norm1(x))
        else:
            attended, x_f = self.attn(self.norm1(x)), None
        x = x + self.drop_path(attended)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, x_f

