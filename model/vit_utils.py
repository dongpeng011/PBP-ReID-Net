# encoding: utf-8
"""Small building blocks shared by the PBP-ReID backbone.

These helpers are kept separate from the paper modules so that the three
contributions (EASA, HPRA, GAL) stay easy to read.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# initialisation helpers
# --------------------------------------------------------------------------- #
def _no_grad_trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Truncated normal initialisation (same as in timm / TransReID)."""

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def resize_pos_embed(posemb, posemb_new, height, width, model_path=None):
    """Interpolate the positional embedding to a different patch grid."""
    ntok_new = posemb_new.shape[1] - 1
    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    print('Resized position embedding from {} to {} with grid {}x{}'.format(
        tuple(posemb.shape), tuple(posemb_new.shape), height, width))
    side = int(round(math.sqrt(posemb_grid.shape[0])))
    if side * side != posemb_grid.shape[0]:
        # already a non-square grid (e.g. 16x8); fall back to the stored shape
        side_h, side_w = None, None
        for candidate_h, candidate_w in ((14, 14), (16, 8), (24, 8), (12, 6), (8, 4)):
            if candidate_h * candidate_w == posemb_grid.shape[0]:
                side_h, side_w = candidate_h, candidate_w
                break
        if side_h is None:
            raise RuntimeError('Cannot infer the patch grid of the positional embedding')
        side = (side_h, side_w)
    if isinstance(side, int):
        grid = posemb_grid.reshape(1, side, side, -1).permute(0, 3, 1, 2)
    else:
        grid = posemb_grid.reshape(1, side[0], side[1], -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(height, width), mode='bilinear', align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, height * width, -1)
    return torch.cat([posemb_token, grid], dim=1)


# --------------------------------------------------------------------------- #
# attention kernel
# --------------------------------------------------------------------------- #
def multi_head_attention(query, key, value, num_heads, scale, dropout_p=0.0):
    """Scaled dot-product attention on ``[B, N, C]`` tensors.

    ``torch.nn.functional.scaled_dot_product_attention`` is used when the
    installed PyTorch provides it (>= 2.0): it fuses softmax and the two matrix
    products, which matters for the many short sequences of HPRA.  The manual
    path is kept for older versions and produces the same values up to
    floating-point noise.

    Returns a tensor of shape ``[B, num_heads, N, head_dim]``.
    """
    batch, num_tokens, channels = query.shape
    head_dim = channels // num_heads
    q = query.reshape(batch, num_tokens, num_heads, head_dim).permute(0, 2, 1, 3)
    k = key.reshape(batch, num_tokens, num_heads, head_dim).permute(0, 2, 1, 3)
    v = value.reshape(batch, num_tokens, num_heads, head_dim).permute(0, 2, 1, 3)

    if hasattr(F, 'scaled_dot_product_attention'):
        # ``scale`` was only added to the fused kernel in PyTorch 2.1; for older
        # versions a custom scale is emulated by rescaling the queries, which is
        # numerically equivalent.
        default_scale = head_dim ** -0.5
        if scale is None or abs(scale - default_scale) < 1e-12:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        q = q * (scale / default_scale)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=True)
    return attn @ v


# --------------------------------------------------------------------------- #
# generic transformer blocks
# --------------------------------------------------------------------------- #
def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Standard full-token multi-head self-attention."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, num_tokens, channels = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        x = multi_head_attention(query, key, value, self.num_heads, self.scale,
                                 self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(batch, num_tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


# --------------------------------------------------------------------------- #
# patch embedding
# --------------------------------------------------------------------------- #
class PatchEmbed(nn.Module):
    """Split the image into patches and project them to ``embed_dim`` channels."""

    def __init__(self, img_size=224, patch_size=16, stride_size=16,
                 in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        stride_size = (stride_size, stride_size) if isinstance(stride_size, int) else tuple(stride_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.num_x = (img_size[1] - patch_size[1]) // stride_size[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size[0] + 1
        self.num_patches = self.num_x * self.num_y
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        print('PatchEmbed: stride {}, patch grid {} x {} = {} tokens'.format(
            stride_size, self.num_y, self.num_x, self.num_patches))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_in = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / fan_in))
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        height, width = x.shape[2], x.shape[3]
        if (height, width) != self.img_size:
            raise ValueError(
                'Input size {}x{} does not match the model size {}x{}'.format(
                    height, width, self.img_size[0], self.img_size[1]))
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
