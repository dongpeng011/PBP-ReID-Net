# encoding: utf-8
"""Build the PBP-ReID model used for training and testing.

The wrapper adds the BNNeck and the identity classifier of the TransReID
family on top of :class:`model.pbp_reid.PBPReID`, so the training code sees the
usual ``(cls_score, global_feat)`` pair in training mode and a single feature
vector in evaluation mode.
"""

import torch
import torch.nn as nn

from model.pbp_reid import PBPReID


def weights_init_kaiming(module):
    classname = module.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if module.affine:
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)


def weights_init_classifier(module):
    classname = module.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(module.weight, std=0.001)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class PBPReIDModel(nn.Module):
    """Backbone + BNNeck + classifier."""

    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(PBPReIDModel, self).__init__()
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.cos_layer = cfg.MODEL.COS_LAYER

        if 'small' in cfg.MODEL.TRANSFORMER_TYPE:
            in_planes = 384
        else:
            in_planes = 768
        self.in_planes = in_planes

        camera = camera_num if cfg.MODEL.SIE_CAMERA else 0
        view = view_num if cfg.MODEL.SIE_VIEW else 0

        self.base = PBPReID(
            img_size=cfg.INPUT.SIZE_TRAIN,
            patch_size=16,
            stride_size=cfg.MODEL.STRIDE_SIZE[0],
            embed_dim=in_planes,
            depth=12,
            num_heads=12 if in_planes == 768 else 6,
            mlp_ratio=4.0,
            qkv_bias=True,
            camera=camera,
            view=view,
            sie_xishu=cfg.MODEL.SIE_COE,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-6),
            part_num=cfg.MODEL.PART_NUM,
            hpra_start=cfg.MODEL.HPRA_START,
            hpra_heads=cfg.MODEL.HPRA_HEADS,
            hpra_rel_heads=cfg.MODEL.HPRA_REL_HEADS,
            hpra_rel_share=cfg.MODEL.HPRA_REL_SHARE,
            hpra_rel_norm=cfg.MODEL.HPRA_REL_NORM,
            use_gal=cfg.MODEL.GAL,
        )
        # the classifier of the backbone is unused: the identity loss is applied
        # after the BNNeck below, as in TransReID
        self.base.fc = nn.Identity()

        if cfg.MODEL.PRETRAIN_CHOICE != 'none':
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH)
            print('PBP-ReID: initialised the backbone from {}'.format(cfg.MODEL.PRETRAIN_PATH))

        self.num_classes = num_classes
        self.classifier = nn.Linear(in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = sum(p.numel() for p in self.base.parameters())
        new_modules = sum(p.numel() for name, p in self.base.named_parameters()
                          if ('qkv_rel' in name or 'proj_rel' in name
                              or name.startswith('gal.')))
        print('PBP-ReID: {:.1f}M trainable parameters in total, {:.1f}M in the '
              'backbone, {:.2f}M in the HPRA/GAL additions'.format(
                  total / 1e6, backbone / 1e6, new_modules / 1e6))

    def forward(self, x, label=None, cam_label=None, view_label=None, **kwargs):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat
        if self.neck_feat == 'after':
            return feat
        return global_feat

    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        print('PBP-ReID: loaded a full model checkpoint from {}'.format(trained_path))


def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.TYPE != 'TR':
        raise NotImplementedError('Only the transformer backbone is implemented, got {}'.format(
            cfg.MODEL.TYPE))
    model = PBPReIDModel(num_class, camera_num, view_num, cfg)
    print(model)
    return model
