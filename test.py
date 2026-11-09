# encoding: utf-8
"""Evaluate a trained PBP-ReID checkpoint.

The datasets that are evaluated are the ones listed under ``DATASETS`` in the
config file, so the same script covers the single-scene and the multi-scene
protocol::

    python test.py --config_file configs/PBP-ReID.yml \
        MODEL.DEVICE_ID "('0')" MODEL.PRETRAIN_CHOICE none \
        TEST.WEIGHT logs/market1501/pbp_reid_60.pth OUTPUT_DIR logs/market1501

Results are printed to the log and written to ``OUTPUT_DIR/test_results.txt``.
"""

import argparse
import json
import os

import torch

from config import cfg
from datasets import make_combine_dataloader
from model import make_model
from processor import do_test
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description='PBP-ReID testing')
    parser.add_argument('--config_file', default='', type=str)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger('PBP-ReID', cfg.OUTPUT_DIR, if_train=False)
    logger.info(args)
    if args.config_file:
        logger.info('loaded configuration file %s', args.config_file)
    logger.info('running with config:\n%s', cfg)

    if not cfg.TEST.WEIGHT:
        raise ValueError('TEST.WEIGHT must point to a trained checkpoint')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    _, val_loader_dict, _, total_classes, len_query_list = make_combine_dataloader(cfg)
    model = make_model(cfg, num_class=total_classes, camera_num=0, view_num=0)

    state_dict = torch.load(cfg.TEST.WEIGHT, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info('checkpoint loaded: %d missing keys, %d unexpected keys',
                len(missing), len(unexpected))
    if missing:
        logger.info('missing keys (first 10): %s', list(missing)[:10])

    results = do_test(cfg, model, val_loader_dict, len_query_list, logger)

    result_path = os.path.join(cfg.OUTPUT_DIR, 'test_results.txt')
    with open(result_path, 'w', encoding='utf-8') as handle:
        handle.write('# PBP-ReID evaluation results\n')
        handle.write('# config: {}\n'.format(args.config_file))
        handle.write('# weight: {}\n'.format(cfg.TEST.WEIGHT))
        for name, metrics in results.items():
            handle.write('{} mAP={:.2f} R1={:.2f} R5={:.2f} R10={:.2f}\n'.format(
                name, 100 * metrics['mAP'], 100 * metrics['R1'],
                100 * metrics['R5'], 100 * metrics['R10']))
        handle.write('\n# json\n')
        handle.write(json.dumps(results, indent=2))
    logger.info('results written to %s', result_path)


if __name__ == '__main__':
    main()
