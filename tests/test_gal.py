# encoding: utf-8
"""Unit tests for GAL, the global-local additive fusion (Sec. 3.3).

Run with:  python -m pytest tests -q
"""

import torch

from model.modules import GlobalLocalAdditiveFusion


def test_fusion_is_an_addition():
    fusion = GlobalLocalAdditiveFusion(dim=32, part_num=3)
    x_global = torch.randn(4, 32)
    x_part = torch.randn(4, 3, 32)
    expected = x_global + fusion.proj(x_part.reshape(4, -1))
    assert torch.allclose(fusion(x_global, x_part), expected)


def test_output_keeps_the_backbone_dimension():
    fusion = GlobalLocalAdditiveFusion(dim=768, part_num=3)
    out = fusion(torch.randn(2, 768), torch.randn(2, 3, 768))
    assert out.shape == (2, 768)


def test_parameter_budget_is_pc2_plus_c():
    dim, part_num = 768, 3
    fusion = GlobalLocalAdditiveFusion(dim=dim, part_num=part_num)
    assert sum(p.numel() for p in fusion.parameters()) == dim * part_num * dim + dim


def test_part_num_is_validated():
    import pytest
    fusion = GlobalLocalAdditiveFusion(dim=32, part_num=3)
    with pytest.raises(ValueError):
        fusion(torch.randn(2, 32), torch.randn(2, 4, 32))


def test_two_dimensional_input_is_accepted():
    fusion = GlobalLocalAdditiveFusion(dim=32, part_num=1)
    out = fusion(torch.randn(2, 32), torch.randn(2, 32))
    assert out.shape == (2, 32)
