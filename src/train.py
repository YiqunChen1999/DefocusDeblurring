
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

from utils import utils

@utils.log_info_wrapper("Start train model.")
@torch.no_grad()
def train_one_epoch(
    epoch: int, 
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    loss_fn, 
    optimizer: torch.optim.Optimizer, 
    lr_scheduler, 
    logger=None, 
    *args, 
    **kwargs, 
):
    model.train()
    # TODO  Prepare to log info.
    log_info = print if logger is None else logger.log_info
    pbar = tqdm(total=len(data_loader))
    total_loss = []
    # TODO  Read data and train and record info.
    with utils.log_info(msg="Train at epoch: {}".format(str(epoch).zfill(3)), level="INFO", state=True, logger=logger):
        for idx, data, anno in enumerate(data_loader):
            optimizer.zero_grad()
            out, loss = utils.inference_and_cal_loss(model=model, inp=data, anno=anno, loss_fn=loss_fn)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.detach().cpu().item())
            pbar.set_description("Epoch: {:<3}, loss: {:<5}".format(epoch, sum(total_loss)/len(total_loss)))
            pbar.update()
        lr_scheduler.step()
        pbar.close()
    # TODO  Return some info.
    raise NotImplementedError("Function train_one_epoch is not implemented yet.")
