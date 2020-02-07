# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import torch
import apex

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []
    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY

    bias_params = []
    bias_lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
    bias_weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if "bias" in key:
            bias_params.append(value)
        else:
            params.append(value)
    # optimizer = torch.optim.SGD(
    optimizer = apex.optimizers.FusedSGD(
        [
            {"params": params, "lr": lr, "weight_decay": weight_decay},
            {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
        ],
        lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
