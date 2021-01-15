
"""
Author:
    Yiqun Chen
Docs:
    Generate results without ground truth.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import utils, metrics

@utils.log_info_wrapper("Start generate results.")
@torch.no_grad()
def generate(
    cfg, 
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    logger=None, 
    *args, 
    **kwargs, 
):
    model.eval()
    # Prepare to log info.
    log_info = print if logger is None else logger.log_info
    total_loss = []
    # Read data and evaluate and record info.
    with utils.log_info(msg="Generate results", level="INFO", state=True, logger=logger):
        pbar = tqdm(total=len(data_loader), dynamic_ncols=True)
        for idx, data in enumerate(data_loader):
            output = utils.inference(model=model, data=data, device=device)

            for i in range(output.shape[0]):
                save_dir = os.path.join(cfg.SAVE.DIR, "test")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                path2file = os.path.join(save_dir, data["img_idx"][i]+"_g.png")
                succeed = utils.save_image(output[i].detach().cpu().numpy(), cfg.DATA.MEAN, cfg.DATA.NORM, path2file)
                if not succeed:
                    log_info("Cannot save image to {}".format(path2file))
            pbar.update()
        pbar.close()

