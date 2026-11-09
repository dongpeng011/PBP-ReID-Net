# encoding: utf-8
"""EASA -- Efficient Anchoring and Structural Alignment (paper, Sec. 3.1).

The module is deliberately *parameter free*: it only computes, once, which
patch-token indices belong to each of the ``P`` coarse vertical body bands
(head, torso, lower body).  Nothing is learned here, which is exactly the
property that removes the pose/segmentation network from part-aware ReID.

For a patch grid with ``num_y`` rows and ``num_x`` columns and ``P`` bands the
first ``P - 1`` bands contain ``floor(num_y / P)`` rows each and the remaining
rows are assigned to the last band, as in Eq. (1)-(3) of the paper::

    N_p = floor(num_y / P) * num_x
    N_i = N_p                      for i < P
    N_P = N - (P - 1) * N_p

With ``num_y = 16`` and ``P = 3`` the bands contain 5, 5 and 6 rows, i.e. 40,
40 and 48 of the 128 tokens.
"""

import torch
import torch.nn as nn


class BodyPriorAnchoring(nn.Module):
    """Row-wise, parameter-free anchoring of patch tokens to body bands.

    Parameters
    ----------
    num_y, num_x : int
        Shape of the patch grid produced by the backbone.
    part_num : int
        Number of coarse body bands ``P``.

    The band indices are registered as buffers, so they follow the model to the
    GPU with ``.to(device)`` and never appear in the parameter count.
    """

    def __init__(self, num_y, num_x, part_num=3):
        super(BodyPriorAnchoring, self).__init__()
        if part_num < 1:
            raise ValueError('part_num must be >= 1, got {}'.format(part_num))
        if part_num > num_y:
            raise ValueError(
                'part_num ({}) cannot exceed the number of patch rows ({})'.format(
                    part_num, num_y))

        self.num_y = int(num_y)
        self.num_x = int(num_x)
        self.part_num = int(part_num)
        self.num_tokens = self.num_y * self.num_x

        rows_per_band = self.num_y // self.part_num
        boundaries = []
        start = 0
        for band in range(self.part_num):
            end = start + rows_per_band if band < self.part_num - 1 else self.num_y
            boundaries.append((start, end))
            start = end
        self.boundaries = boundaries

        # one LongTensor per band, kept as buffers -> no parameters, moves with .to()
        for band, (row_start, row_end) in enumerate(boundaries):
            rows = torch.arange(row_start, row_end, dtype=torch.long)
            cols = torch.arange(self.num_x, dtype=torch.long)
            # row-major flattening of the patch grid: index = row * num_x + col
            indices = (rows.unsqueeze(1) * self.num_x + cols.unsqueeze(0)).reshape(-1)
            self.register_buffer('band_indices_{}'.format(band), indices, persistent=False)
        # inverse mapping: token -> band id, useful for visualisation / debugging
        token_to_band = torch.zeros(self.num_tokens, dtype=torch.long)
        for band, (row_start, row_end) in enumerate(boundaries):
            token_to_band[row_start * self.num_x:row_end * self.num_x] = band
        self.register_buffer('token_to_band', token_to_band, persistent=False)

    # ------------------------------------------------------------------ API
    def band_indices(self, device=None):
        """Return the list of per-band token indices (``P`` LongTensors)."""
        indices = [getattr(self, 'band_indices_{}'.format(band))
                   for band in range(self.part_num)]
        if device is not None:
            indices = [index.to(device) for index in indices]
        return indices

    def band_sizes(self):
        """Number of tokens in each band (``N_i`` of the paper)."""
        return [int(index.numel()) for index in self.band_indices()]

    def split(self, tokens):
        """Split ``[B, N, C]`` tokens into ``P`` tensors of ``[B, N_i, C]``."""
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(
                'Expected {} tokens, got {}'.format(self.num_tokens, tokens.shape[1]))
        return [tokens.index_select(1, index) for index in self.band_indices(tokens.device)]

    @staticmethod
    def merge(bands):
        """Inverse of :meth:`split`; the bands are contiguous slices, so a plain
        concatenation restores the original row-major token order."""
        return torch.cat(bands, dim=1)

    def extra_repr(self):
        return 'num_y={}, num_x={}, part_num={}, band_sizes={}'.format(
            self.num_y, self.num_x, self.part_num, self.band_sizes())
