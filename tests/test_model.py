# encoding: utf-8
"""End-to-end tests of the assembled PBP-ReID model.

These tests build the real ViT-Base model on CPU, so they take a few seconds
each.  Run with:  python -m pytest tests -q
"""

import pytest
import torch
import torch.nn as nn

from config import cfg
from model import make_model


@pytest.fixture(scope='module')
def base_config():
    config = cfg.clone()
    config.defrost()
    config.merge_from_list(['MODEL.PRETRAIN_CHOICE', 'none', 'INPUT.SIZE_TRAIN', '[256, 128]'])
    config.freeze()
    return config


def _variant(base_config, overrides):
    config = base_config.clone()
    config.defrost()
    config.merge_from_list(overrides)
    config.freeze()
    return config


def test_train_and_eval_forward(base_config):
    model = make_model(base_config, num_class=10, camera_num=0, view_num=0)
    model.eval()
    images = torch.randn(2, 3, 256, 128)
    with torch.no_grad():
        feature = model(images)
    assert feature.shape == (2, 768)
    assert torch.isfinite(feature).all()

    model.train()
    cls_score, global_feat = model(images, torch.tensor([0, 1]))
    assert cls_score.shape == (2, 10)
    assert global_feat.shape == (2, 768)


def test_backward_pass(base_config):
    model = make_model(base_config, num_class=10, camera_num=0, view_num=0)
    model.train()
    images = torch.randn(2, 3, 256, 128)
    labels = torch.tensor([0, 1])
    cls_score, global_feat = model(images, labels)
    loss = nn.functional.cross_entropy(cls_score, labels) + global_feat.square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_ablation_switches_build(base_config):
    variants = {
        'vit_base_reference': ['MODEL.HPRA_START', '12', 'MODEL.GAL', 'False'],
        'hpra_only': ['MODEL.GAL', 'False'],
        'gal_only': ['MODEL.HPRA_START', '12'],
        'separate_level2': ['MODEL.HPRA_REL_SHARE', 'False'],
        'all_hpra_blocks': ['MODEL.HPRA_START', '0'],
    }
    images = torch.randn(2, 3, 256, 128)
    for name, overrides in variants.items():
        model = make_model(_variant(base_config, overrides), num_class=10,
                           camera_num=0, view_num=0)
        model.eval()
        with torch.no_grad():
            out = model(images)
        assert out.shape == (2, 768), name


def test_fusion_identity_holds_in_the_model(base_config):
    model = make_model(base_config, num_class=10, camera_num=0, view_num=0)
    model.eval()
    images = torch.randn(2, 3, 256, 128)
    with torch.no_grad():
        fused, x_f = model.base(images, return_parts=True)
        tokens, _ = model.base.forward_features(images)
        manual = tokens[:, 1:].mean(dim=1) + model.base.gal.proj(x_f.reshape(x_f.size(0), -1))
    assert torch.allclose(fused, manual, atol=1e-4)


def test_reference_model_has_no_hpra_parameters(base_config):
    model = make_model(_variant(base_config, ['MODEL.HPRA_START', '12', 'MODEL.GAL', 'False']),
                       num_class=10, camera_num=0, view_num=0)
    names = [n for n, _ in model.named_parameters()]
    assert not any('qkv_rel' in n or 'gal.' in n for n in names)


def test_invalid_hpra_start_is_rejected(base_config):
    with pytest.raises(ValueError):
        make_model(_variant(base_config, ['MODEL.HPRA_START', '13']),
                   num_class=10, camera_num=0, view_num=0)
