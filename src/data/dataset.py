
"""
Author:
    Yiqun Chen
Docs:
    Dataset classes.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

from utils import utils

_DATASET = {}

def add_dataset(dataset):
    _DATASET[dataset.__name__] = dataset
    return dataset


@add_dataset
class DualPixelCanon(torch.utils.data.Dataset):
    def __init__(self, cfg, split, *args, **kwargs):
        super(DualPixelCanon, self).__init__()
        self.cfg = cfg
        self.split = split
        self._build()

    def _build(self):
        raise NotImplementedError("Method DualPixelCanon._build is not implemented yet.")

    def _preprocess(self):
        raise NotImplementedError("Method DualPixelCanon._preprocess is not implemented yet.")

    def __getitem__(self, idx):
        raise NotImplementedError("Method DualPixelCanon.__getitem__ is not implemented yet.")

    def __len__(self):
        raise NotImplementedError("Method DualPixelCanon.__len__ is not implemented yet.")


@add_dataset
class DualPixelNTIRE2021(torch.utils.data.Dataset):
    def __init__(self, cfg, split, *args, **kwargs):
        super(DualPixelNTIRE2021, self).__init__()
        self.cfg = cfg
        self.split = split
        self._build()

    def _build(self):
        raise NotImplementedError("Method DualPixelNTIRE2021._build is not implemented yet.")

    def _preprocess(self):
        raise NotImplementedError("Method DualPixelNTIRE2021._preprocess is not implemented yet.")

    def __getitem__(self, idx):
        raise NotImplementedError("Method DualPixelNTIRE2021.__getitem__ is not implemented yet.")

    def __len__(self):
        raise NotImplementedError("Method DualPixelNTIRE2021.__len__ is not implemented yet.")

@add_dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split, *args, **kwargs):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self._build()

    def _build(self):
        raise NotImplementedError("Dataset is not implemeted yet.")

    def __len__(self):
        raise NotImplementedError("Dataset is not implemeted yet.")

    def __getitem__(self, idx):
        raise NotImplementedError("Dataset is not implemeted yet.")