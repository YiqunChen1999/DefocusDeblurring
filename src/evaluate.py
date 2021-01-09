
"""
Author:
    Yiqun Chen
Docs:
    Functions to train a model.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import utils, metrics

@utils.log_info_wrapper("Start evaluate model.")
@torch.no_grad()
def evaluate(
    epoch: int, 
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    loss_fn, 
    logger=None, 
    save=False, 
    *args, 
    **kwargs, 
):
    model.eval()
    # TODO  Prepare to log info.
    log_info = print if logger is None else logger.log_info
    pbar = tqdm(total=len(data_loader))
    total_loss = []
    # TODO  Read data and evaluate and record info.
    with utils.log_info(msg="Evaluate at epoch: {}".format(str(epoch).zfill(3)), level="INFO", state=True, logger=logger):
        log_info("Will{}save results to {}".format(" " if save else " not ", cfg.SAVE.DIR))
        for idx, data in enumerate(data_loader):
            out, loss = utils.inference_and_cal_loss(model=model, data=data, loss_fn=loss_fn, device=device)
            total_loss.append(loss.detach().cpu().item())
            if save:
                # TODO Save results to directory.
                pass
            pbar.set_description("Epoch: {:<3}, loss: {:<5}".format(epoch, sum(total_loss)/len(total_loss)))
            pbar.update()
        pbar.close()
    # TODO  Return some info.
    # raise NotImplementedError("Function evaluate is not implemented yet.")

