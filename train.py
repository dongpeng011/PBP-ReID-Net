# encoding: utf-8
"""Train PBP-ReID.

Examples
--------
Single-scene regime (Table 2 of the paper)::

    python train.py --config_file configs/PBP-ReID.yml \
        MODEL.DEVICE_ID "('0')" OUTPUT_DIR logs/market1501

Multi-scene regime (Table 3)::

    python train.py --config_file configs/PBP-ReID-multiscene.yml \
        MODEL.DEVICE_ID "('0')" OUTPUT_DIR logs/multiscene

Every option of ``config/defaults.py`` can be overridden from the command line,
which is how the ablations of Table 5 are reproduced, e.g.::

    python train.py --config_file configs/PBP-ReID.yml MODEL.HPRA_START 12 MODEL.GAL False
"""

import argparse
import os
import random

import numpy as np
import torch

from config import cfg
from datasets import make_combine_dataloader
from loss import make_loss
from model import make_model
from processor import do_train
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.logger import setup_logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description='PBP-ReID training')
    parser.add_argument('--config_file', default='', type=str,
                        help='path to the yaml config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='overrides of config options, e.g. MODEL.DEVICE_ID "(\'0\')"')
    parser.add_argument('--local_rank', default=0, type=int)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        raise NotImplementedError('Distributed training is not used by PBP-ReID')

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger('PBP-ReID', cfg.OUTPUT_DIR, if_train=True)
    logger.info('saving models and logs to %s', cfg.OUTPUT_DIR)
    logger.info(args)

    if args.config_file:
        logger.info('loaded configuration file %s', args.config_file)
        with open(args.config_file, 'r', encoding='utf-8-sig') as handle:
            logger.info('\n' + handle.read())
    logger.info('running with config:\n%s', cfg)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, val_loader_dict, _, total_classes, len_query_list = \
        make_combine_dataloader(cfg)
    logger.info('training set: %d images, %d identities, %d datasets',
                len(train_loader.dataset), total_classes, len(val_loader_dict))

    model = make_model(cfg, num_class=total_classes, camera_num=0, view_num=0)
    loss_func, center_criterion = make_loss(cfg, num_classes=total_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    do_train(cfg, model, center_criterion, train_loader, val_loader_dict,
             optimizer, optimizer_center, scheduler, loss_func, len_query_list,
             args.local_rank)


if __name__ == '__main__':
    main()

