# encoding: utf-8
"""Count parameters and estimate FLOPs of the compared models.

The FLOPs are counted analytically with the usual convention
``FLOPs = 2 x MACs`` for matrix multiplications and convolutions:

* patch embedding: ``H * W * C_in * k^2 * C_out / stride^2`` MACs
* linear layers and MLP: ``tokens * in_features * out_features`` MACs
* attention score and value product: ``2 * tokens^2 * C`` MACs for the standard
  block, ``sum_i 2 * N_i^2 * C`` for the intra-band level of HPRA, plus
  ``2 * P^2 * C`` for the inter-band level

Run::

    python tools/count_params_flops.py
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg  # noqa: E402
from model import make_model  # noqa: E402
from model.modules import HPRAttention  # noqa: E402
from model.vit_utils import Attention, Mlp, PatchEmbed  # noqa: E402


def linear_macs(module, tokens):
    return tokens * module.in_features * module.out_features


def count_macs(module, tokens_per_sample, part_num=3):
    """Analytic MAC count of one forward pass for a single image."""
    macs = 0
    for child in module.modules():
        if isinstance(child, PatchEmbed):
            height, width = child.img_size
            kernel = child.patch_size
            macs += (height * width * child.proj.in_channels
                     * kernel[0] * kernel[1] * child.proj.out_channels
                     // (child.stride_size[0] * child.stride_size[1]))
        elif isinstance(child, nn.Linear):
            # counted below through its parent block to know the token count
            continue
        elif isinstance(child, Mlp):
            macs += linear_macs(child.fc1, tokens_per_sample)
            macs += linear_macs(child.fc2, tokens_per_sample)
        elif isinstance(child, HPRAttention):
            # level 1: shared projections applied band by band, tokens in total
            macs += linear_macs(child.qkv, tokens_per_sample)
            macs += linear_macs(child.proj, tokens_per_sample)
            # level 2: P descriptors
            if child.rel_share:
                macs += linear_macs(child.qkv, part_num) + linear_macs(child.proj, part_num)
            else:
                macs += linear_macs(child.qkv_rel, part_num) + linear_macs(child.proj_rel, part_num)
            macs += 2 * part_num * part_num * child.dim
        elif isinstance(child, Attention):
            macs += linear_macs(child.qkv, tokens_per_sample)
            macs += linear_macs(child.proj, tokens_per_sample)
            macs += 2 * tokens_per_sample * tokens_per_sample * child.qkv.in_features
    return macs


def intra_band_macs(band_sizes, dim):
    """Attention score + value MACs of the intra-band level of HPRA."""
    return sum(2 * size * size * dim for size in band_sizes)


def describe(name, cfg_override, num_class=751):
    config = cfg.clone()
    config.defrost()
    options = []
    for key, value in cfg_override.items():
        options += [key, value]
    config.merge_from_list(options)
    config.freeze()
    model = make_model(config, num_class=num_class, camera_num=0, view_num=0)
    backbone = model.base

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in backbone.parameters())

    tokens = backbone.num_patches + 1          # + class token
    tokens_per_block = tokens
    macs = 0
    # patch embedding
    height, width = backbone.patch_embed.img_size
    kernel = backbone.patch_embed.patch_size
    macs += (height * width * 3 * kernel[0] * kernel[1] * backbone.embed_dim
             // (backbone.patch_embed.stride_size[0] * backbone.patch_embed.stride_size[1]))
    # blocks
    standard_blocks = backbone.hpra_start
    hpra_blocks = backbone.depth - backbone.hpra_start
    for block in backbone.blocks:
        macs += linear_macs(block.attn.qkv, tokens_per_block)
        macs += linear_macs(block.attn.proj, tokens_per_block)
        macs += linear_macs(block.mlp.fc1, tokens_per_block)
        macs += linear_macs(block.mlp.fc2, tokens_per_block)
        if isinstance(block.attn, HPRAttention):
            macs += intra_band_macs(backbone.anchoring.band_sizes(), backbone.embed_dim)
            macs += 2 * backbone.part_num * backbone.part_num * backbone.embed_dim
            if not block.attn.rel_share:
                macs += linear_macs(block.attn.qkv_rel, backbone.part_num)
                macs += linear_macs(block.attn.proj_rel, backbone.part_num)
        else:
            macs += 2 * tokens_per_block * tokens_per_block * backbone.embed_dim
    if backbone.gal is not None:
        macs += backbone.gal.proj.in_features * backbone.gal.proj.out_features

    flops = 2 * macs / 1e9
    print('{:<44s} {:>10.2f} {:>10.2f} {:>9.2f}'.format(
        name, params / 1e6, backbone_params / 1e6, flops))
    return {'name': name, 'params_m': params / 1e6,
            'backbone_params_m': backbone_params / 1e6, 'flops_g': flops}


def main():
    base = {'MODEL.PRETRAIN_CHOICE': 'none', 'INPUT.SIZE_TRAIN': [256, 128]}
    variants = [
        ('ViT-B reference (no HPRA, no GAL)',
         dict(base, **{'MODEL.HPRA_START': 12, 'MODEL.GAL': False})),
        ('PBP-ReID, separate level-2 projections',
         dict(base, **{'MODEL.HPRA_REL_SHARE': False})),
        ('PBP-ReID, shared level-2 projections',
         dict(base, **{'MODEL.HPRA_REL_SHARE': True})),
        ('HPRA only (no GAL)', dict(base, **{'MODEL.GAL': False})),
        ('GAL only (no HPRA)', dict(base, **{'MODEL.HPRA_START': 12})),
    ]
    print('{:<44s} {:>10s} {:>10s} {:>9s}'.format(
        'model', 'params(M)', 'backbone(M)', 'FLOPs(G)'))
    print('-' * 78)
    results = [describe(name, override) for name, override in variants]
    print('-' * 78)
    reference = results[0]
    for result in results[1:]:
        print('{:<44s} {:+.2f} M, {:+.1f}% FLOPs vs. the ViT-B reference'.format(
            result['name'],
            result['params_m'] - reference['params_m'],
            100 * (result['flops_g'] / reference['flops_g'] - 1)))


if __name__ == '__main__':
    main()

