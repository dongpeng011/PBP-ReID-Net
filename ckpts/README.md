# Pre-trained backbones

Put the ImageNet (or self-supervised) ViT-Base checkpoint here, for example:

```
ckpts/
└── vit_base_patch16_224.pth
```

then either keep the default in the config files

```yaml
MODEL:
  PRETRAIN_CHOICE: 'imagenet'
  PRETRAIN_PATH: 'ckpts/vit_base_patch16_224.pth'
```

or point to it from the command line:

```bash
python train.py --config_file configs/PBP-ReID.yml MODEL.PRETRAIN_PATH /path/to/vit_base.pth
```

The loader accepts checkpoints wrapped in `{'model': ...}` or
`{'state_dict': ...}` and keys with the `module.` / `base.` prefixes, copies
every tensor whose name and shape match, interpolates the positional embedding
when the patch grid differs, and initialises the layers that are new in PBP-ReID
(the inter-band projections and `W_f`) from scratch.

Checkpoints are **not** distributed with this repository.  `MODEL.PRETRAIN_CHOICE
none` trains from random initialisation.
