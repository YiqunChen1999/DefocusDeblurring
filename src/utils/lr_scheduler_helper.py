
"""
Author  Yiqun Chen
Docs    Help build lr scheduler.
"""

import os, sys, warnings
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

_SCHEDULER = {}

def add_scheduler(scheduler):
    _SCHEDULER[scheduler.__name__] = scheduler
    return scheduler


@add_scheduler
class StepLRScheduler:
    def __init__(self, cfg, optimizer: torch.optim.Optimizer, *args, **kwargs):
        super(StepLRScheduler, self).__init__()
        self.cfg = cfg
        self.optimizer = optimizer
        self._build()

    def _build(self):
        self.update_epoch = list(self.cfg.SCHEDULER.UPDATE_EPOCH)
        self.update_scale = self.cfg.SCHEDULER.UPDATE_SCALE
        # NOTE scheduler.step() should be called after optimizer.step(), thus cnt start from 1.
        self.cnt = 1

    def update(self):
        old_lrs = []
        new_lrs = []
        if self.cnt in self.update_epoch:
            for param_group in self.optimizer.param_groups:
                old_lrs.append(param_group["lr"])
                new_lrs.append(old_lrs[-1]*self.update_scale)
                assert len(old_lrs) == len(new_lrs)
                if new_lrs[-1] <= 0:
                    warnings.warn("Learning rate {} is not larger than 0.0.".format(new_lrs[-1]))
                param_group["lr"] = new_lrs[-1]
        self.cnt += 1

    def step(self):
        self.update()

    def sychronize(self, epoch):
        self.cnt = epoch + 1


@add_scheduler
class LinearLRScheduler:
    def __init__(self, cfg, optimizer, *args, **kwargs):
        super(LinearLRScheduler, self).__init__()
        self.cfg = cfg
        self.optimizer = optimizer
        self._build()

    def _build(self):
        raise NotImplementedError("Linear LR Scheduler is not implemented yet.")

    def update(self):
        raise NotImplementedError("Linear LR Scheduler is not implemented yet.")

    def step(self):
        raise NotImplementedError("Linear LR Scheduler is not implemented yet.")


def build_scheduler(cfg, optimizer):
    return _SCHEDULER[cfg.SCHEDULER.SCHEDULER](cfg, optimizer)