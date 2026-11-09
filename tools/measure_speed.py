# encoding: utf-8
"""Measure parameters, FLOPs, latency and throughput of PBP-ReID and of the
ViT-Base reference, i.e. the numbers reported in Table 4 and plotted in Fig. 12
of the paper.

    python tools/measure_speed.py --config_file configs/PBP-ReID.yml --batch-size 64

The latency is the per-image forward time inside one batch (the convention used
in the paper), measured after warm-up with CUDA events.  Run it on the same GPU
you report in the paper.
"""

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg  # noqa: E402
from model import make_model  # noqa: E402
from count_params_flops import count_macs  # noqa: E402


def build(config, num_class):
    config = config.clone()
    config.defrost()
    config.merge_from_list(['MODEL.PRETRAIN_CHOICE', 'none'])
    config.freeze()
    return make_model(config, num_class=num_class, camera_num=0, view_num=0)


def analytic_flops(model):
    backbone = model.base
    tokens = backbone.num_patches + 1
    macs = count_macs(backbone, tokens, backbone.part_num)
    return 2 * macs / 1e9


@torch.no_grad()
def measure_latency(model, batch_size, iterations, warmup, device):
    images = torch.randn(batch_size, 3, 256, 128, device=device)
    model.eval()
    for _ in range(warmup):
        model(images)
    torch.cuda.synchronize() if device.type == 'cuda' else None

    samples = []
    for _ in range(iterations):
        if device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(images)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            import time
            begin = time.perf_counter()
            model(images)
            samples.append((time.perf_counter() - begin) * 1000.0)
    per_batch = statistics.median(samples)
    return per_batch / batch_size, 1000.0 * batch_size / per_batch


def main():
    parser = argparse.ArgumentParser(description='measure efficiency of PBP-ReID')
    parser.add_argument('--config_file', default='configs/PBP-ReID.yml')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--num-classes', type=int, default=751,
                        help='size of the identity classifier (751 for Market-1501)')
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        print('WARNING: no GPU found, timings will not be comparable with the paper')

    variants = [
        ('ViT-B reference (no HPRA, no GAL)',
         ['MODEL.HPRA_START', '12', 'MODEL.GAL', 'False']),
        ('PBP-ReID (paper configuration)', []),
        ('PBP-ReID, separate level-2 projections', ['MODEL.HPRA_REL_SHARE', 'False']),
    ]

    rows = []
    for name, overrides in variants:
        config = cfg.clone()
        config.defrost()
        if overrides:
            config.merge_from_list(overrides)
        config.freeze()
        model = build(config, args.num_classes).to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        backbone_params = sum(p.numel() for p in model.base.parameters()) / 1e6
        flops = analytic_flops(model)
        latency, fps = measure_latency(model, args.batch_size, args.iterations,
                                       args.warmup, device)
        rows.append((name, params, backbone_params, flops, latency, fps))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print('\n{:<42s} {:>9s} {:>9s} {:>9s} {:>10s} {:>7s}'.format(
        'model', 'params(M)', 'bbone(M)', 'FLOPs(G)', 'lat(ms)', 'FPS'))
    print('-' * 92)
    for name, params, backbone_params, flops, latency, fps in rows:
        print('{:<42s} {:>9.2f} {:>9.2f} {:>9.2f} {:>10.3f} {:>7.0f}'.format(
            name, params, backbone_params, flops, latency, fps))
    print('-' * 92)
    print('batch size {}, {} timed iterations after {} warm-up iterations'.format(
        args.batch_size, args.iterations, args.warmup))
    print('FLOPs follow the 2 x MACs convention and cover one 256x128 crop.')


if __name__ == '__main__':
    main()
