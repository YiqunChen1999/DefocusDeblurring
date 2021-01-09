
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
    """
    Info:
        Dataset of Canon, folders and files structure:
        DualPixelCanon:
            - train:
                - source:
                    00000.png
                    00001.png
                    ...
                - target:
                    00000.png
                    00001.png
                    ...
                - l_view:
                    00000.png
                    00001.png
                    ...
                - r_view:
                    00000.png
                    00001.png
                    ...
            - valid:
                ...
            - test:
                ...
    """
    def __init__(self, cfg, split, *args, **kwargs):
        super(DualPixelCanon, self).__init__()
        self.cfg = cfg
        self.split = split
        self._build()

    def _build(self):
        self.data = []
        self.path2imgs = os.path.join(self.cfg.DATA.DIR[self.cfg.DATA.DATASET], self.split)
        self.s_img_list = os.listdir(self.path2imgs, "source")
        self.t_img_list = os.listdir(self.path2imgs, "target")
        self.l_img_list = os.listdir(self.path2imgs, "l_view")
        self.r_img_list = os.listdir(self.path2imgs, "r_view")

        for idx, gt_img in enumerate(self.t_img_list):
            img_idx = gt_img.split(".")[0]
            if img_idx not in self.s_img_list[idx] or img_idx not in self.l_img_list[idx] or img_idx not in self.r_img_list[idx]:
                continue
            self.data.append({
                "img_idx": img_idx, 
                "target": gt_img, 
                "source": self.s_img_list[idx], 
                "l_view": self.l_img_list[idx], 
                "r_view": self.r_img_list[idx], 
            })
        self._preprocess()
        raise NotImplementedError("Method DualPixelCanon._build is not implemented yet.")

    def _preprocess(self):
        raise NotImplementedError("Method DualPixelCanon._preprocess is not implemented yet.")

    def __getitem__(self, idx):
        target = np.array(Image.open(os.path.join(self.path2imgs, "target", self.data[idx]["target"])))
        source = np.array(Image.open(os.path.join(self.path2imgs, "source", self.data[idx]["source"])))
        l_img = np.array(Image.open(os.path.join(self.path2imgs, "l_view", self.data[idx]["l_view"])))
        r_img = np.array(Image.open(os.path.join(self.path2imgs, "r_view", self.data[idx]["r_view"])))

        data = {}
        # Transform: [H, W, C] -> [C, H, W]
        data["target"] = np.transpose(target, (2, 0, 1))
        data["source"] = np.transpose(source, (2, 0, 1))
        data["l_view"] = np.transpose(l_img, (2, 0, 1))
        data["r_view"] = np.transpose(r_img, (2, 0, 1))
        return data
        raise NotImplementedError("Method DualPixelCanon.__getitem__ is not implemented yet.")

    def __len__(self):
        return len(self.data)
        raise NotImplementedError("Method DualPixelCanon.__len__ is not implemented yet.")


@add_dataset
class DualPixelNTIRE2021(torch.utils.data.Dataset):
    """
    Info:
        Dataset of NTIRE2021, folders and files structure:
        DualPixelNTIRE2021:
            - train:
                - target:
                    00000.png
                    00001.png
                    ...
                - l_view:
                    00000.png
                    00001.png
                    ...
                - r_view:
                    00000.png
                    00001.png
                ...
            - valid:
                ...
            - test:
                ...
    """
    def __init__(self, cfg, split, *args, **kwargs):
        super(DualPixelNTIRE2021, self).__init__()
        self.cfg = cfg
        self.split = split
        self._build()

    def _build(self):
        self.data = []
        self.path2imgs = os.path.join(self.cfg.DATA.DIR[self.cfg.DATA.DATASET], self.split)
        t_img_list = os.listdir(os.path.join(self.path2imgs, "target"))
        t_img_dict = {img.split(".")[0]: img for img in t_img_list}
        l_img_list = os.listdir(os.path.join(self.path2imgs, "l_view"))
        l_img_dict = {img.split(".")[0]: img for img in l_img_list}
        r_img_list = os.listdir(os.path.join(self.path2imgs, "r_view"))
        r_img_dict = {img.split(".")[0]: img for img in r_img_list}
        for idx, img_idx in enumerate(t_img_dict):
            if img_idx in l_img_dict.keys() and img_idx in r_img_dict.keys():
                self.data.append({
                    "img_idx": img_idx, 
                    "target": t_img_dict[img_idx], 
                    "l_view": l_img_dict[img_idx], 
                    "r_view": r_img_dict[img_idx], 
                })
        # self._preprocess()
        # raise NotImplementedError("Method DualPixelNTIRE2021._build is not implemented yet.")

    def _preprocess(self):
        raise ValueError("Some images in training set is broken, please rectify them.")
        raise NotImplementedError("Method DualPixelNTIRE2021._preprocess is not implemented yet.")

    def __getitem__(self, idx):
        data = {}
        target = np.array(Image.open(os.path.join(self.path2imgs, "target", self.data[idx]["target"])), dtype=np.float32)
        l_img = np.array(Image.open(os.path.join(self.path2imgs, "l_view", self.data[idx]["l_view"])), dtype=np.float32)
        r_img = np.array(Image.open(os.path.join(self.path2imgs, "r_view", self.data[idx]["r_view"])), dtype=np.float32)

        # Transform: [H, W, C] -> [C, H, W]
        data["target"] = np.transpose(target, (2, 0, 1))
        data["l_view"] = np.transpose(l_img, (2, 0, 1))
        data["r_view"] = np.transpose(r_img, (2, 0, 1))

        return data
        
    def __len__(self):
        return len(self.data)
        

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


if __name__ == "__main__":
    from configs.configs import cfg

    dataset = DualPixelNTIRE2021(cfg, "train")
    for i in range(len(dataset)):
        item = dataset[i]
        