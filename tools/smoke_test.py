# encoding: utf-8
"""Self-contained smoke test: builds the model, runs forward/backward on CPU and
(if available) GPU, and checks the algebraic identities of EASA / HPRA / GAL.

    python tools/smoke_test.py

It needs no dataset and no pretrained checkpoint, so it is the fastest way to
verify that the code is installed correctly.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg  # noqa: E402
from model import make_model  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print('  [ok] {}'.format(message))


def main():
    cfg.merge_from_file(os.path.join('configs', 'PBP-ReID.yml'))
    cfg.merge_from_list(['MODEL.PRETRAIN_CHOICE', 'none', 'SOLVER.MAX_EPOCHS', '1'])
    cfg.freeze()

    print('\n=== 1. build the model ===')
    model = make_model(cfg, num_class=10, camera_num=0, view_num=0)
    backbone = model.base
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('  trainable parameters: {:.2f} M'.format(total / 1e6))

    print('\n=== 2. EASA band anchoring ===')
    sizes = backbone.anchoring.band_sizes()
    check(sum(sizes) == backbone.num_patches,
          'bands cover all {} tokens: {}'.format(backbone.num_patches, sizes))
    check(sizes[:-1] == [sizes[0]] * (len(sizes) - 1),
          'all but the last band have the same size')
    merged = backbone.anchoring.merge(backbone.anchoring.split(
        torch.arange(backbone.num_patches).reshape(1, -1, 1).float()))
    check(torch.equal(merged.reshape(-1).long(), torch.arange(backbone.num_patches)),
          'split + merge restores the row-major token order')
    check(all(not p.requires_grad for p in backbone.anchoring.parameters()),
          'EASA has no trainable parameters')

    print('\n=== 3. forward / backward ===')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    images = torch.randn(4, 3, 256, 128, device=device)
    labels = torch.arange(4, device=device) % 10

    model.train()
    cls_score, global_feat = model(images, labels)
    check(cls_score.shape == (4, 10), 'classifier output is [4, 10]')
    check(global_feat.shape == (4, 768), 'global feature is [4, 768]')
    loss = nn.functional.cross_entropy(cls_score, labels) + global_feat.square().mean()
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    check(grad_norm > 0, 'gradients reach the parameters (sum |grad| = {:.3f})'.format(grad_norm))

    model.eval()
    with torch.no_grad():
        feature = model(images)
    check(feature.shape == (4, 768), 'evaluation feature is [4, 768]')
    check(torch.isfinite(feature).all(), 'evaluation feature is finite')

    print('\n=== 4. GAL identity  F = GAP(X) + W_f * X_f ===')
    with torch.no_grad():
        fused, x_f = backbone(images, return_parts=True)
        tokens, _ = backbone.forward_features(images)
        manual = tokens[:, 1:].mean(dim=1) + backbone.gal.proj(x_f.reshape(x_f.shape[0], -1))
    check(torch.allclose(fused, manual, atol=1e-5),
          'F equals the additive fusion of the paper (max diff {:.2e})'.format(
              (fused - manual).abs().max().item()))
    check(x_f.shape == (4, backbone.part_num, 768), 'X_f is [B, P, C]')

    print('\n=== 5. ablations used in Table 5 ===')
    for name, overrides in (
            ('ViT-B reference (no HPRA, no GAL)', ['MODEL.HPRA_START', '12', 'MODEL.GAL', 'False']),
            ('HPRA only', ['MODEL.GAL', 'False']),
            ('GAL only (no HPRA)', ['MODEL.HPRA_START', '12'])):
        variant = cfg.clone()
        variant.defrost()
        variant.merge_from_list(overrides)
        variant.freeze()
        variant_model = make_model(variant, num_class=10, camera_num=0, view_num=0)
        variant_model.to(device).eval()
        with torch.no_grad():
            out = variant_model(images)
        check(out.shape == (4, 768), '{} -> [4, 768]'.format(name))

    print('\n=== 6. loading a ViT-Base checkpoint (pos-embed resize + new layers) ===')
    reference_cfg = cfg.clone()
    reference_cfg.defrost()
    reference_cfg.merge_from_list(['MODEL.PRETRAIN_CHOICE', 'none',
                                   'MODEL.HPRA_START', '12', 'MODEL.GAL', 'False',
                                   'INPUT.SIZE_TRAIN', '[224, 224]'])
    reference_cfg.freeze()
    reference_model = make_model(reference_cfg, num_class=751, camera_num=0, view_num=0)
    fake_checkpoint = {'module.base.' + key: value
                       for key, value in reference_model.base.state_dict().items()}
    checkpoint_path = os.path.join('logs', 'fake_vit_base.pth')
    os.makedirs('logs', exist_ok=True)
    torch.save(fake_checkpoint, checkpoint_path)

    pretrained_cfg = cfg.clone()
    pretrained_cfg.defrost()
    pretrained_cfg.merge_from_list(['MODEL.PRETRAIN_CHOICE', 'imagenet',
                                    'MODEL.PRETRAIN_PATH', checkpoint_path])
    pretrained_cfg.freeze()
    pretrained_model = make_model(pretrained_cfg, num_class=10, camera_num=0, view_num=0)
    check(pretrained_model.base.pos_embed.shape[1] == backbone.num_patches + 1,
          'pos_embed was interpolated to {} tokens'.format(backbone.num_patches + 1))
    same_qkv = torch.allclose(
        pretrained_model.base.blocks[0].attn.qkv.weight,
        reference_model.base.blocks[0].attn.qkv.weight)
    check(same_qkv, 'pretrained attention weights were transferred to the first block')
    same_hpra_qkv = torch.allclose(
        pretrained_model.base.blocks[-1].attn.qkv.weight,
        reference_model.base.blocks[-1].attn.qkv.weight)
    check(same_hpra_qkv, 'pretrained attention weights were transferred to an HPRA block')
    pretrained_model.to(device)
    pretrained_model.eval()
    with torch.no_grad():
        out = pretrained_model(images)
    check(out.shape == (4, 768), 'the pretrained model runs, feature [4, 768]')
    pretrained_model.train()
    cls_score, _ = pretrained_model(images, labels)
    check(cls_score.shape == (4, 10), 'the pretrained model runs, logits [4, 10]')
    os.remove(checkpoint_path)

    print('\nall checks passed on {}\n'.format(device))


if __name__ == '__main__':
    main()


