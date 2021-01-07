
"""
Author  Yiqun Chen
Docs    Help build lr scheduler.
"""

import os, sys
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