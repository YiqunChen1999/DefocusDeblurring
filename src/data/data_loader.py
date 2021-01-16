
"""
Author:
    Yiqun Chen
Docs:
    Help build data loader.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import utils
from .dataset import _DATASET


def build_dataset(cfg, dataset, split):
    return _DATASET[dataset](cfg, split)
    
def build_data_loader(cfg, dataset, split):
    shuffle = True if split == "train" else False
    num_workers = cfg.DATA.NUMWORKERS
    dataset = build_dataset(cfg, dataset, split)
    if split == "test" or (cfg.DATA.DATASET == "DualPixelNTIRE2021" and split == "valid"):
        batch_size = 2
    else:
        batch_size = cfg.GENERAL.BATCH_SIZE
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )
    return data_loader