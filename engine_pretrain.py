# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    epoch_size=1):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    data_loader_i = iter(data_loader)
    for data_iter_step in metric_logger.log_every(range(epoch_size), print_freq, header):
        (batch, _) = next(data_loader_i)
        # we use a per iteration (instead of per epoch) lr scheduler
        if isinstance(batch, tuple):
            samples, visual_tokens = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = visual_tokens.to(device, non_blocking=True)
        else: # hack for consistency
            samples = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = samples

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            loss_dict, _, _ = model(samples, visual_tokens, mask_ratio=args.mask_ratio)

        loss = torch.stack([loss_dict[l] for l in loss_dict if 'unscaled' not in l]).sum()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(model, data_loader, device, epoch, log_writer, args):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, visual_tokens = batch
        samples = samples.to(device, non_blocking=True)
        visual_tokens = visual_tokens.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss_dict, _, _ = model(samples, visual_tokens, mask_ratio=args.mask_ratio)

        loss = torch.stack([loss_dict[l] for l in loss_dict if 'unscaled' not in l]).sum()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats for val:", metric_logger)
    return {'val_' + k: meter.global_avg for k, meter in metric_logger.meters.items()}
