# encoding: utf-8
"""PBP-ReID backbone: ViT-Base + EASA + HPRA + GAL.

The class in this file is the complete visual encoder of the paper.  It is
deliberately independent of the head/loss code so that it can be used on its
own::

    from model.pbp_reid import PBPReID
    backbone = PBPReID(img_size=(256, 128))
    feature = backbone(images)                 # [B, 768]

Forward pass (see Fig. 2 of the paper):

1. ``PatchEmbed`` -> ``X`` of ``N = num_y * num_x`` tokens, prepend the class
   token, add the positional (and optionally camera/view) embedding;
2. the first ``hpra_start`` blocks use full-token self-attention, the remaining
   blocks use HPRA (Sec. 3.2) with the bands produced by EASA (Sec. 3.1);
3. the last HPRA block returns the band-relation representation ``X_f``;
4. ``GAL`` sums ``GAP(X)`` and ``Flatten(X_f) W_f + b_f`` (Sec. 3.3).

Setting ``hpra_start = depth`` and ``use_gal = False`` turns the model into the
plain ViT-Base reference used in Table 4 of the paper; setting
``hpra_start = 0`` applies HPRA to every block.
"""

import torch
import torch.nn as nn

from model.modules import BodyPriorAnchoring, GlobalLocalAdditiveFusion, PBPBlock
from model.vit_utils import PatchEmbed, resize_pos_embed, trunc_normal_


class PBPReID(nn.Module):
    """Vision transformer with body-prior anchoring, HPRA and additive fusion."""

    def __init__(self, img_size=(256, 128), patch_size=16, stride_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm, camera=0, view=0,
                 sie_xishu=1.0, local_feature=False,
                 part_num=3, hpra_start=6, hpra_heads=12, hpra_rel_heads=4,
                 hpra_rel_share=False, hpra_rel_norm=True, use_gal=True):
        super(PBPReID, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.local_feature = local_feature

        # ---------------------------------------------------------- backbone
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      stride_size=stride_size, in_chans=in_chans,
                                      embed_dim=embed_dim)
        self.num_y = self.patch_embed.num_y
        self.num_x = self.patch_embed.num_x
        self.num_patches = self.patch_embed.num_patches

        # ------------------------------------------------- body prior (EASA)
        self.part_num = int(part_num)
        self.anchoring = BodyPriorAnchoring(self.num_y, self.num_x, self.part_num)
        self.band_indices = self.anchoring.band_indices()

        # -------------------------------------------------------- HPRA range
        self.depth = int(depth)
        self.hpra_start = int(hpra_start)
        if not 0 <= self.hpra_start <= self.depth:
            raise ValueError('hpra_start must be in [0, depth], got {}'.format(self.hpra_start))
        self.num_hpra_blocks = self.depth - self.hpra_start

        print('PBP-ReID: patch grid {} x {} = {} tokens, {} body bands {}'.format(
            self.num_y, self.num_x, self.num_patches, self.part_num,
            self.anchoring.band_sizes()))
        print('PBP-ReID: HPRA in blocks [{}, {}) ({} of {} blocks), GAL={}'.format(
            self.hpra_start, self.depth, self.num_hpra_blocks, self.depth, bool(use_gal)))

        # -------------------------------------------------------- tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ------------------------------------------- camera / view embedding
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
        else:
            self.sie_embed = None

        # ------------------------------------------------------ transformer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList([
            PBPBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[index],
                norm_layer=norm_layer,
                hpra=index >= self.hpra_start,
                part_num=self.part_num,
                band_indices=self.band_indices,
                rel_heads=hpra_rel_heads,
                rel_share=hpra_rel_share,
                rel_norm=hpra_rel_norm,
                num_prefix=1)
            for index in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # -------------------------------------------------- GAL (Sec. 3.3)
        self.use_gal = bool(use_gal)
        self.gal = GlobalLocalAdditiveFusion(embed_dim, self.part_num) if self.use_gal else None

        # ------------------------------------------------------------ head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    # ------------------------------------------------------------------ init
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'sie_embed'}

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # --------------------------------------------------------------- forward
    def forward_features(self, x, cam_label=None, view_label=None):
        """Return the final token sequence ``[B, 1 + N, C]``."""
        batch = x.shape[0]
        tokens = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1)
        x = x + self.pos_embed

        if self.sie_embed is not None and cam_label is not None:
            if self.cam_num > 0 and self.view_num > 0:
                x = x + self.sie_xishu * self.sie_embed[cam_label * self.view_num + view_label]
            elif self.cam_num > 0:
                x = x + self.sie_xishu * self.sie_embed[cam_label]
            elif self.view_num > 0:
                x = x + self.sie_xishu * self.sie_embed[view_label]

        x = self.pos_drop(x)

        x_f = None
        for block in self.blocks:
            x, block_x_f = block(x)
            if block_x_f is not None:
                x_f = block_x_f
        x = self.norm(x)
        return x, x_f

    def forward(self, x, cam_label=None, view_label=None, return_parts=False):
        """Return the identity descriptor ``F`` ``[B, C]`` of Eq. (11).

        With ``return_parts=True`` the band descriptors ``X_f`` are returned as
        well, which is useful for the attention visualisations of Fig. 13.
        """
        tokens, x_f = self.forward_features(x, cam_label=cam_label, view_label=view_label)

        if self.local_feature:
            return tokens

        # x_g = GAP(X) over the patch tokens (the class token is excluded)
        x_global = tokens[:, 1:].mean(dim=1)

        if self.use_gal:
            if x_f is None:
                # No HPRA block in this configuration (the "+ GAL only" row of
                # the ablation): the band descriptors are obtained by pooling
                # the anchored bands of the final token sequence, which is the
                # parameter-free half of Eq. (7).
                bands = self.anchoring.split(tokens[:, 1:])
                x_f = torch.stack([band.mean(dim=1) for band in bands], dim=1)
            feature = self.gal(x_global, x_f)
        else:
            feature = x_global

        if return_parts:
            return feature, x_f
        return feature

    # ------------------------------------------------------------- loading
    def load_param(self, model_path=None, param_dict=None):
        """Load a ViT(-B) checkpoint, skipping the layers that are new in PBP-ReID.

        The ``qkv`` / ``proj`` weights of HPRA keep the names of the plain
        attention layer, so the pretrained attention of every block is reused;
        the inter-band projection and ``W_f`` of GAL are initialised here and
        reported as skipped keys.
        """
        if param_dict is None:
            if model_path is None:
                raise ValueError('Either model_path or param_dict must be given')
            param_dict = torch.load(model_path, map_location='cpu')
        for wrapper in ('model', 'state_dict'):
            if wrapper in param_dict and isinstance(param_dict[wrapper], dict):
                param_dict = param_dict[wrapper]
                break

        state = self.state_dict()
        loaded, skipped = 0, []
        for key, value in param_dict.items():
            key = key.replace('module.', '')
            if key.startswith('base.'):
                key = key[len('base.'):]
            if 'head' in key or 'dist' in key or key.startswith('classifier'):
                skipped.append(key)
                continue
            if 'patch_embed.proj.weight' in key and value.dim() < 4:
                out_channels, in_channels, height, width = self.patch_embed.proj.weight.shape
                value = value.reshape(out_channels, -1, height, width)
            elif key == 'pos_embed' and value.shape != state[key].shape:
                value = resize_pos_embed(value, state[key], self.num_y, self.num_x, model_path)
            if key not in state:
                skipped.append(key)
                continue
            if state[key].shape != value.shape:
                skipped.append(key)
                continue
            with torch.no_grad():
                state[key].copy_(value)
            loaded += 1

        print('PBP-ReID: loaded {} parameter tensors from the checkpoint, '
              'kept {} new/ignored tensors'.format(loaded, len(skipped)))
        if skipped:
            preview = ', '.join(skipped[:6])
            if len(skipped) > 6:
                preview += ', ...'
            print('PBP-ReID: example skipped keys: {}'.format(preview))

    def count_parameters(self, trainable_only=True):
        """Number of parameters, useful to compare with Table 4 of the paper."""
        parameters = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)
