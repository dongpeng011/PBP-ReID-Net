# encoding: utf-8
"""Default configuration of PBP-ReID.

Only the options that are actually consumed by the code are kept here.  The
three modules of the paper are controlled by the ``MODEL.PART_*``,
``MODEL.HPRA_*`` and ``MODEL.GAL`` switches, so that the ablations of
Table 5 can be reproduced from the command line without editing any file.
"""

try:  # prefer the real yacs when it is installed
    from yacs.config import CfgNode as CN
except ImportError:  # pragma: no cover - exercised on machines without yacs
    from config.cfg_node import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.NAME = 'pbp_reid'

# ImageNet / self-supervised ViT-Base checkpoint used to initialise the backbone.
_C.MODEL.PRETRAIN_PATH = 'ckpts/vit_base_patch16_224.pth'
# 'imagenet' | 'self' | 'none'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# ---------------------------------------------------------------- backbone
_C.MODEL.TYPE = 'TR'
_C.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224_PBPReID'
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0

# ------------------------------------------------- body prior (EASA, Sec. 3.1)
# Number of coarse vertical bands P.  Three bands = head / torso / lower body.
_C.MODEL.PART_NUM = 3

# ------------------------------------- hierarchical part-relation attn (HPRA)
# Blocks [HPRA_START, depth) replace full-token self-attention by HPRA, which
# is what reduces the attention cost of Table 1 / Table 4.  Set HPRA_START to
# the depth (12) to obtain the plain ViT-Base reference row.
_C.MODEL.HPRA_START = 6
# heads of the intra-band attention (must divide the embedding dimension)
_C.MODEL.HPRA_HEADS = 12
# heads of the compact inter-band attention over the P band descriptors
_C.MODEL.HPRA_REL_HEADS = 4
# share the QKV/output projections of the two attention levels
_C.MODEL.HPRA_REL_SHARE = True
# layer normalisation applied to the band descriptors before level 2
_C.MODEL.HPRA_REL_NORM = True

# ------------------------------ global-local additive fusion (GAL, Sec. 3.3)
_C.MODEL.GAL = True

# ---------------------------------------------------------------- BNNeck head
_C.MODEL.NECK = 'bnneck'
_C.MODEL.COS_LAYER = False
_C.MODEL.IF_WITH_CENTER = 'no'
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
_C.MODEL.NO_MARGIN = False
_C.MODEL.IF_LABELSMOOTH = 'on'
_C.MODEL.DIST_TRAIN = False

# camera / viewpoint side information (SIE of TransReID); off by default
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# -----------------------------------------------------------------------------
# INPUT   (the paper uses 256 x 128 crops -> 16 x 8 = 128 tokens)
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = [256, 128]
_C.INPUT.SIZE_TEST = [256, 128]
_C.INPUT.PROB = 0.5
_C.INPUT.RE_PROB = 0.5
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.PADDING = 10
_C.INPUT.GCJ_PROB = 0.0
_C.INPUT.GGS_PROB = 0.0
_C.INPUT.GB_PROB = 0.0
_C.INPUT.LCJ_PROB = 0.0
_C.INPUT.LGS_PROB = 0.0
_C.INPUT.LB_PROB = 0.0
_C.INPUT.AUTO_AUG = False

# -----------------------------------------------------------------------------
# DATASETS
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# one entry per training subset; a single entry gives the single-scene regime
# of Table 2, several entries give the multi-scene regime of Table 3
_C.DATASETS.NAMES = ('combine')
_C.DATASETS.ROOTS = ['data/market1501']
_C.DATASETS.COMBINE_NAMES = ['market1501']
# scene id of each subset, e.g. [1] for Market-1501, [1, 1] for Market+MSMT17,
# 0 = low resolution, 2 = clothing change, 3 = occlusion, 4 = infrared
_C.DATASETS.COMBINE_TYPE = [1]
# True: identities of different subsets are disjoint (unified label space)
_C.DATASETS.COMBINE_PID = True

# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.SAMPLER = 'softmax_triplet'
_C.DATALOADER.COMBINE_SAMPLER = 'softmax_triplet'
_C.DATALOADER.NUM_INSTANCE = 4

# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.MAX_EPOCHS = 60
_C.SOLVER.BASE_LR = 3.5e-4
_C.SOLVER.MIN_LR = 0.0
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.BIAS_LR_FACTOR = 1
_C.SOLVER.SEED = 1234
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.MARGIN = 0.3
_C.SOLVER.CENTER_LR = 0.5
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (40, 70)
_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30
_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 50
_C.SOLVER.EVAL_PERIOD = 10
_C.SOLVER.IMS_PER_BATCH = 64

# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.RE_RANKING = False
_C.TEST.WEIGHT = ""
_C.TEST.NECK_FEAT = 'after'
_C.TEST.FEAT_NORM = 'yes'
_C.TEST.DIST_MAT = "dist_mat.npy"
_C.TEST.EVAL = True
# number of times the evaluation is repeated with different seeds (Sec. 4.5)
_C.TEST.REPEAT = 1

# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = "logs/PBP-ReID"

