
"""
Author:
    Yiqun Chen
Docs:
    Organize datasets.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
from tqdm import tqdm
import numpy as np

from utils import utils

split = "valid"

'''path2src = "/home/yqchen/models/defocus-deblurring-dual-pixel/file_names/test_src.npy"
path2trg = "/home/yqchen/models/defocus-deblurring-dual-pixel/file_names/test_trg.npy"
path2test = "/home/yqchen/Data/DualPixelCanon/test"

images_src = np.load(path2src)
images_trg = np.load(path2trg)

assert images_src.shape == images_trg.shape, "ShapeError"
assert os.path.exists(path2test)

target_list = sorted(os.listdir(os.path.join(path2test, "target")))
target_dict = {img.split(".")[0]: img for img in target_list}
source_list = sorted(os.listdir(os.path.join(path2test, "source")))
source_dict = {img.split(".")[0]: img for img in source_list}
l_view_list = sorted(os.listdir(os.path.join(path2test, "l_view")))
l_view_dict = {img.split("_")[0]: img for img in l_view_list}
r_view_list = sorted(os.listdir(os.path.join(path2test, "r_view")))
r_view_dict = {img.split("_")[0]: img for img in r_view_list}

for idx in range(images_src.shape[0]):
    src_idx = images_src[idx]
    trg_idx = images_trg[idx]
    target = target_dict[trg_idx]
    source = source_dict[src_idx]
    l_view = l_view_dict[src_idx]
    r_view = r_view_dict[src_idx]
    os.system("mv {} {}".format(
        os.path.join(path2test, "target", target), os.path.join(path2test, "target", str(idx).zfill(5)+".png")
    ))
    os.system("mv {} {}".format(
        os.path.join(path2test, "source", source), os.path.join(path2test, "source", str(idx).zfill(5)+".png")
    ))
    os.system("mv {} {}".format(
        os.path.join(path2test, "l_view", l_view), os.path.join(path2test, "l_view", str(idx).zfill(5)+".png")
    ))
    os.system("mv {} {}".format(
        os.path.join(path2test, "r_view", r_view), os.path.join(path2test, "r_view", str(idx).zfill(5)+".png")
    ))

'''

src = "/home/yqchen/Data/dd_dp_dataset_validation_inputs_only"
des = "/home/yqchen/Data/DualPixelNTIRE2021"

utils.try_make_path_exists(os.path.join(des, "train", "target"))
utils.try_make_path_exists(os.path.join(des, "train", "source"))
utils.try_make_path_exists(os.path.join(des, "train", "l_view"))
utils.try_make_path_exists(os.path.join(des, "train", "r_view"))

utils.try_make_path_exists(os.path.join(des, "valid", "target"))
utils.try_make_path_exists(os.path.join(des, "valid", "source"))
utils.try_make_path_exists(os.path.join(des, "valid", "l_view"))
utils.try_make_path_exists(os.path.join(des, "valid", "r_view"))

utils.try_make_path_exists(os.path.join(des, "test", "target"))
utils.try_make_path_exists(os.path.join(des, "test", "source"))
utils.try_make_path_exists(os.path.join(des, "test", "l_view"))
utils.try_make_path_exists(os.path.join(des, "test", "r_view"))

img_list = sorted(os.listdir(src))
img_dict = {img.split(".")[0]: img for img in img_list}

pbar = tqdm(total=len(img_list))
for img_idx, img in img_dict.items():
    
    if "l" in img_idx:
        os.system("cp {} {}".format(os.path.join(src, img), os.path.join(des, split, "l_view", img_idx.split("_")[0]+".png")))
    if "r" in img_idx:
        os.system("cp {} {}".format(os.path.join(src, img), os.path.join(des, split, "r_view", img_idx.split("_")[0]+".png")))
    if "g" in img_idx:
        os.system("cp {} {}".format(os.path.join(src, img), os.path.join(des, split, "target", img_idx.split("_")[0]+".png")))
    pbar.update()