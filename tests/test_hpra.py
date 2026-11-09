# encoding: utf-8
"""Unit tests for HPRA, the hierarchical part-relation attention (Sec. 3.2).

Run with:  python -m pytest tests -q
"""

import torch

from model.modules import BodyPriorAnchoring, HPRAttention
from model.vit_utils import Attention


def _modules(dim=96, heads=4, rel_heads=2, part_num=3, num_prefix=1):
    anchoring = BodyPriorAnchoring(num_y=16, num_x=8, part_num=part_num)
    hpra = HPRAttention(dim=dim, num_heads=heads, part_num=part_num,
                        band_indices=anchoring.band_indices(), rel_heads=rel_heads,
                        num_prefix=num_prefix)
    return anchoring, hpra


def test_shapes_and_prefix_preservation():
    anchoring, hpra = _modules()
    x = torch.randn(2, 1 + anchoring.num_tokens, 96)
    out, x_f = hpra(x)
    assert out.shape == x.shape
    assert x_f.shape == (2, anchoring.part_num, 96)
    # the class token bypasses both attention levels
    assert torch.allclose(out[:, :1], x[:, :1], atol=1e-6)


def test_level1_matches_explicit_per_band_attention():
    """Level 1 must be identical to running the shared projection + MHSA per band."""
    torch.manual_seed(0)
    anchoring, hpra = _modules()
    hpra.eval()
    x = torch.randn(2, 1 + anchoring.num_tokens, 96)
    body = x[:, 1:]

    with torch.no_grad():
        out, _ = hpra(x)
        qkv = hpra.qkv(body)
        q, k, v = qkv.chunk(3, dim=-1)
        expected = []
        for index in anchoring.band_indices():
            qb, kb, vb = (t.index_select(1, index) for t in (q, k, v))
            head_dim = 96 // hpra.num_heads
            qh = qb.reshape(2, -1, hpra.num_heads, head_dim).permute(0, 2, 1, 3)
            kh = kb.reshape(2, -1, hpra.num_heads, head_dim).permute(0, 2, 1, 3)
            vh = vb.reshape(2, -1, hpra.num_heads, head_dim).permute(0, 2, 1, 3)
            attn = (qh @ kh.transpose(-2, -1)) * hpra.scale
            attended = (attn.softmax(dim=-1) @ vh).transpose(1, 2).reshape(2, -1, 96)
            expected.append(attended + body.index_select(1, index))
        expected = hpra.proj(torch.cat(expected, dim=1))

    assert torch.allclose(out[:, 1:], expected, atol=2e-5)


def test_attention_score_entries_drop_by_two_thirds():
    """P bands of N/P tokens have P * (N/P)^2 = N^2/P score entries."""
    anchoring = BodyPriorAnchoring(num_y=16, num_x=8, part_num=3)
    full = anchoring.num_tokens ** 2
    banded = sum(size ** 2 for size in anchoring.band_sizes())
    assert banded / full < 0.35


def test_shared_projections_add_no_level2_parameters():
    anchoring = BodyPriorAnchoring(num_y=16, num_x=8, part_num=3)
    shared = HPRAttention(dim=96, num_heads=4, part_num=3,
                          band_indices=anchoring.band_indices(), rel_heads=2, rel_share=True)
    separate = HPRAttention(dim=96, num_heads=4, part_num=3,
                            band_indices=anchoring.band_indices(), rel_heads=2, rel_share=False)
    shared_params = sum(p.numel() for p in shared.parameters())
    separate_params = sum(p.numel() for p in separate.parameters())
    # the separate variant adds a 3C x C QKV projection and a C x C output
    # projection, both with bias: 4C^2 weights + 4C biases
    dim = 96
    assert separate_params - shared_params == 4 * dim * dim + 4 * dim


def test_matches_standard_attention_parameter_names():
    """A ViT-B checkpoint must still load into an HPRA block."""
    anchoring = BodyPriorAnchoring(num_y=16, num_x=8, part_num=3)
    hpra = HPRAttention(dim=96, num_heads=4, part_num=3, band_indices=anchoring.band_indices())
    plain = Attention(dim=96, num_heads=4)
    assert hpra.qkv.weight.shape == plain.qkv.weight.shape
    assert hpra.proj.weight.shape == plain.proj.weight.shape
