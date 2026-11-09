# encoding: utf-8
"""Unit tests for EASA, the body-prior anchoring module (Sec. 3.1).

Run with:  python -m pytest tests -q
"""

import torch

from model.modules import BodyPriorAnchoring


def test_band_sizes_follow_the_paper():
    """N_i = floor(N_y / P) * N_x for i < P, and the remainder goes to the last band."""
    anchoring = BodyPriorAnchoring(num_y=16, num_x=8, part_num=3)
    assert anchoring.band_sizes() == [40, 40, 48]
    assert sum(anchoring.band_sizes()) == 128


def test_bands_partition_every_token_exactly_once():
    anchoring = BodyPriorAnchoring(num_y=16, num_x=8, part_num=3)
    indices = torch.cat(anchoring.band_indices())
    assert indices.numel() == anchoring.num_tokens
    assert torch.equal(torch.sort(indices).values, torch.arange(anchoring.num_tokens))


def test_split_and_merge_are_inverse():
    anchoring = BodyPriorAnchoring(num_y=16, num_x=8, part_num=3)
    tokens = torch.arange(128, dtype=torch.float32).reshape(1, 128, 1)
    merged = anchoring.merge(anchoring.split(tokens))
    assert torch.equal(merged, tokens)


def test_no_parameters():
    anchoring = BodyPriorAnchoring(num_y=16, num_x=8, part_num=3)
    assert sum(p.numel() for p in anchoring.parameters()) == 0


def test_other_configurations():
    for num_y, num_x, part_num in ((14, 14, 3), (24, 8, 2), (16, 8, 4), (12, 6, 3)):
        anchoring = BodyPriorAnchoring(num_y=num_y, num_x=num_x, part_num=part_num)
        sizes = anchoring.band_sizes()
        assert sum(sizes) == num_y * num_x
        assert len(sizes) == part_num
        # all bands but the last have the same number of rows
        rows = [size // num_x for size in sizes]
        assert len(set(rows[:-1])) <= 1 and rows[-1] >= rows[0]


def test_invalid_partition_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        BodyPriorAnchoring(num_y=2, num_x=8, part_num=3)
