# Verification log

Environment: Python 3.8.8, torch 2.0.0+cu117, GPU NVIDIA GeForce RTX 3090
--- 1) compileall ---
compileall exit code: 0 (0 = every file compiles)
--- 2) dummy data ---
dummy market dataset written to data\dummy_market (20 identities, 140 files)
dummy msmt dataset written to data\dummy_msmt (20 identities, 124 files)
--- 3) smoke test ---
  [ok] bands cover all 128 tokens: [40, 40, 48]
  [ok] all but the last band have the same size
  [ok] split + merge restores the row-major token order
  [ok] EASA has no trainable parameters
  [ok] classifier output is [4, 10]
  [ok] global feature is [4, 768]
  [ok] gradients reach the parameters (sum |grad| = 68243.663)
  [ok] evaluation feature is [4, 768]
  [ok] evaluation feature is finite
  [ok] F equals the additive fusion of the paper (max diff 4.77e-07)
  [ok] X_f is [B, P, C]
  [ok] ViT-B reference (no HPRA, no GAL) -> [4, 768]
  [ok] HPRA only -> [4, 768]
  [ok] GAL only (no HPRA) -> [4, 768]
  [ok] pos_embed was interpolated to 129 tokens
  [ok] pretrained attention weights were transferred to the first block
  [ok] pretrained attention weights were transferred to an HPRA block
  [ok] the pretrained model runs, feature [4, 768]
  [ok] the pretrained model runs, logits [4, 10]
all checks passed on cuda
--- 4) single-scene training, 2 epochs on synthetic Market-1501 ---
2026-11-09 09:33:51,238 PBP-ReID.train INFO: Epoch 1 done in 3.6s, mean loss 9.988, mean acc 0.038
2026-11-09 09:33:51,647 PBP-ReID.train INFO: saved checkpoint logs\single\pbp_reid_1.pth
2026-11-09 09:33:52,035 PBP-ReID.train INFO: Validation - market1501: mAP 49.8%, Rank-1 35.0%, Rank-5 60.0%, Rank-10 80.0%
2026-11-09 09:33:52,040 PBP-ReID.train INFO: Validation - mean over 1 dataset(s): mAP 49.8%, Rank-1 35.0%
2026-11-09 09:33:53,527 PBP-ReID.train INFO: Epoch 2 done in 1.5s, mean loss 5.840, mean acc 0.013
2026-11-09 09:33:53,918 PBP-ReID.train INFO: saved checkpoint logs\single\pbp_reid_2.pth
2026-11-09 09:33:54,298 PBP-ReID.train INFO: Validation - market1501: mAP 50.8%, Rank-1 35.0%, Rank-5 75.0%, Rank-10 85.0%
2026-11-09 09:33:54,303 PBP-ReID.train INFO: Validation - mean over 1 dataset(s): mAP 50.8%, Rank-1 35.0%
--- 5) test.py loading the checkpoint ---
2026-11-09 09:33:59,329 PBP-ReID INFO: checkpoint loaded: 0 missing keys, 0 unexpected keys
2026-11-09 09:34:02,141 PBP-ReID INFO: Test - market1501: mAP 50.8%, Rank-1 35.0%, Rank-5 75.0%, Rank-10 85.0%
2026-11-09 09:34:02,145 PBP-ReID INFO: Test - mean over 1 dataset(s): mAP 50.8%, Rank-1 35.0%
2026-11-09 09:34:02,147 PBP-ReID INFO: results written to logs\single\test_results.txt
--- 6) multi-scene training, 1 epoch, two datasets, non-contiguous scene ids [0,3] ---
2026-11-09 09:34:05,405 PBP-ReID INFO: training set: 160 images, 40 identities, 2 datasets
2026-11-09 09:34:12,176 PBP-ReID.train INFO: Epoch 1 done in 5.0s, mean loss 8.786, mean acc 0.013
2026-11-09 09:34:12,590 PBP-ReID.train INFO: saved checkpoint logs\multi\pbp_reid_1.pth
2026-11-09 09:34:12,942 PBP-ReID.train INFO: Validation - market1501: mAP 67.4%, Rank-1 60.0%, Rank-5 80.0%, Rank-10 90.0%
2026-11-09 09:34:13,292 PBP-ReID.train INFO: Validation - msmt17: mAP 64.5%, Rank-1 55.0%, Rank-5 75.0%, Rank-10 90.0%
2026-11-09 09:34:13,297 PBP-ReID.train INFO: Validation - mean over 2 dataset(s): mAP 65.9%, Rank-1 57.5%
--- 7) efficiency measurement ---
GPU: NVIDIA GeForce RTX 3090
model                                      params(M)  bbone(M)  FLOPs(G)    lat(ms)     FPS
--------------------------------------------------------------------------------------------
ViT-B reference (no HPRA, no GAL)              86.32     85.75     22.68      1.518     659
PBP-ReID (paper configuration)                 88.10     87.53     22.46      1.545     647
PBP-ReID, separate level-2 projections        102.28    101.70     22.46      1.552     644
--------------------------------------------------------------------------------------------
batch size 64, 50 timed iterations after 20 warm-up iterations
