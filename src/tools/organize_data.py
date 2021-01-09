
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

from utils import utils

split = "train"

src = "/mnt/g/Datasets/dd_dp_dataset_{}".format(split)
des = "/mnt/g/Datasets/DualPixelNTIRE2021"

utils.try_make_path_exists(os.path.join(des, "train", "target"))
utils.try_make_path_exists(os.path.join(des, "train", "source"))
utils.try_make_path_exists(os.path.join(des, "train", "l_img"))
utils.try_make_path_exists(os.path.join(des, "train", "r_img"))

utils.try_make_path_exists(os.path.join(des, "valid", "target"))
utils.try_make_path_exists(os.path.join(des, "valid", "source"))
utils.try_make_path_exists(os.path.join(des, "valid", "l_img"))
utils.try_make_path_exists(os.path.join(des, "valid", "r_img"))

utils.try_make_path_exists(os.path.join(des, "test", "target"))
utils.try_make_path_exists(os.path.join(des, "test", "source"))
utils.try_make_path_exists(os.path.join(des, "test", "l_img"))
utils.try_make_path_exists(os.path.join(des, "test", "r_img"))

img_list = os.listdir(src)

pbar = tqdm(total=len(img_list))
for idx, img in enumerate(img_list):
    if "l" in img.split(".")[0]:
        os.system("cp {} {}".format(os.path.join(src, img), os.path.join(des, split, "l_img")))
    if "r" in img.split(".")[0]:
        os.system("cp {} {}".format(os.path.join(src, img), os.path.join(des, split, "r_img")))
    if "g" in img.split(".")[0]:
        os.system("cp {} {}".format(os.path.join(src, img), os.path.join(des, split, "target")))
    pbar.update()