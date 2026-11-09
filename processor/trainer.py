# encoding: utf-8
"""Training and evaluation loops.

The loops are intentionally small: one epoch of ``CE + triplet`` training over
the combined (or single) training set, followed by a periodic evaluation on the
query/gallery split of every dataset involved.
"""

import logging
import os
import time

import torch
from torch.cuda import amp

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval


def resolve_device(cfg):
    """Return the device to use, falling back to CPU when CUDA is unavailable."""
    requested = str(getattr(cfg.MODEL, 'DEVICE', 'cuda')).lower()
    if requested.startswith('cuda') and not torch.cuda.is_available():
        print('PBP-ReID: CUDA is not available, running on CPU')
        return torch.device('cpu')
    return torch.device(requested if requested != 'cuda' else 'cuda')


def build_evaluators(cfg, len_query_list):
    """One evaluator per dataset, with the dataset-specific PRCC/SYSU protocol."""
    evaluators = {}
    for index, name in enumerate(cfg.DATASETS.COMBINE_NAMES):
        evaluators[name] = R1_mAP_eval(len_query_list[index], max_rank=50,
                                       feat_norm=cfg.TEST.FEAT_NORM, dataset=name)
    return evaluators


@torch.no_grad()
def evaluate(cfg, model, val_loader_dict, evaluators, device, logger=None,
             log_prefix='Validation'):
    """Evaluate every dataset and return ``{name: {'mAP':.., 'R1':..}}``."""
    model.eval()
    results = {}
    for name, loader in val_loader_dict.items():
        evaluator = evaluators[name]
        evaluator.reset()
        for batch in loader:
            img, pid, camid, camids, vid, target_view, _ = batch
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        results[name] = {'mAP': float(mAP), 'R1': float(cmc[0]),
                         'R5': float(cmc[4]), 'R10': float(cmc[9])}
        if logger is not None:
            logger.info('%s - %s: mAP %.1f%%, Rank-1 %.1f%%, Rank-5 %.1f%%, Rank-10 %.1f%%',
                        log_prefix, name, 100 * mAP, 100 * cmc[0], 100 * cmc[4], 100 * cmc[9])
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if results and logger is not None:
        mean_map = sum(v['mAP'] for v in results.values()) / len(results)
        mean_r1 = sum(v['R1'] for v in results.values()) / len(results)
        logger.info('%s - mean over %d dataset(s): mAP %.1f%%, Rank-1 %.1f%%',
                    log_prefix, len(results), 100 * mean_map, 100 * mean_r1)
    return results


def do_train(cfg, model, center_criterion, train_loader, val_loader_dict, optimizer,
             optimizer_center, scheduler, loss_fn, len_query_list, local_rank=0):
    """Train ``model`` for ``cfg.SOLVER.MAX_EPOCHS`` epochs."""
    logger = logging.getLogger('PBP-ReID.train')
    device = resolve_device(cfg)
    model.to(device)

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    epochs = cfg.SOLVER.MAX_EPOCHS

    evaluators = build_evaluators(cfg, len_query_list)
    use_amp = device.type == 'cuda'
    scaler = amp.GradScaler(enabled=use_amp)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    history = []

    logger.info('start training on %s for %d epochs (%d iterations per epoch)',
                device, epochs, len(train_loader))

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        model.train()
        scheduler.step(epoch)

        for iteration, (img, pid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
            if center_criterion is not None:
                optimizer_center.zero_grad()

            img = img.to(device)
            target = pid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=use_amp):
                cls_score, global_feat = model(img, target, cam_label=target_cam,
                                               view_label=target_view)
                loss = loss_fn(cls_score, global_feat, target, target_cam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE and center_criterion is not None:
                for param in center_criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            accuracy = (cls_score.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(accuracy, 1.0)

            if (iteration + 1) % log_period == 0:
                logger.info('Epoch[%d] Iter[%d/%d] loss %.3f, acc %.3f, lr %.2e',
                            epoch, iteration + 1, len(train_loader), loss_meter.avg,
                            acc_meter.avg, scheduler._get_lr(epoch)[0])

        elapsed = time.time() - start_time
        logger.info('Epoch %d done in %.1fs, mean loss %.3f, mean acc %.3f',
                    epoch, elapsed, loss_meter.avg, acc_meter.avg)
        history.append({'epoch': epoch, 'loss': loss_meter.avg, 'acc': float(acc_meter.avg)})

        if checkpoint_period > 0 and epoch % checkpoint_period == 0:
            checkpoint_path = os.path.join(cfg.OUTPUT_DIR,
                                           '{}_{}.pth'.format(cfg.MODEL.NAME, epoch))
            torch.save(model.state_dict(), checkpoint_path)
            logger.info('saved checkpoint %s', checkpoint_path)

        if cfg.TEST.EVAL and eval_period > 0 and epoch % eval_period == 0:
            results = evaluate(cfg, model, val_loader_dict, evaluators, device, logger)
            history[-1]['eval'] = results
            model.train()

    return history


def do_test(cfg, model, val_loader_dict, len_query_list, logger=None):
    """Evaluate a trained model and return the per-dataset metrics."""
    device = resolve_device(cfg)
    model.to(device)
    evaluators = build_evaluators(cfg, len_query_list)
    return evaluate(cfg, model, val_loader_dict, evaluators, device, logger,
                    log_prefix='Test')
