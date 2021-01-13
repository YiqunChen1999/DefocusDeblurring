
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
@torch.enable_grad()
def train_one_epoch(
    epoch: int, 
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    loss_fn, 
    optimizer: torch.optim.Optimizer, 
    lr_scheduler, 
    metrics_logger, 
    logger=None, 
    *args, 
    **kwargs, 
):
    model.train()
    # TODO  Prepare to log info.
    log_info = print if logger is None else logger.log_info
    total_loss = []
    # TODO  Read data and train and record info.
    with utils.log_info(msg="Train at epoch: {}".format(str(epoch).zfill(3)), level="INFO", state=True, logger=logger):
        pbar = tqdm(total=len(data_loader), dynamic_ncols=True)
        for idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            out, loss = utils.inference_and_cal_loss(model=model, data=data, loss_fn=loss_fn, device=device)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.detach().cpu().item())

            metrics_logger.record("train", epoch, "loss", loss.detach().cpu().item())
            output = out.detach().cpu()
            target = data["target"]
            utils.cal_and_record_metrics("train", epoch, output, target, metrics_logger, logger=logger)

            pbar.set_description("Epoch: {:<3}, avg loss: {:<5}, cur loss: {:<5}".format(epoch, round(sum(total_loss)/len(total_loss), 5), round(total_loss[-1], 5)))
            pbar.update()
        lr_scheduler.step()
        pbar.close()
    mean_metrics = metrics_logger.mean("train", epoch)
    log_info("SSIM: {:<5}, PSNR: {:<5}, MAE: {:<5}, Loss: {:<5}".format(
        mean_metrics["SSIM"], mean_metrics["PSNR"], mean_metrics["MAE"], mean_metrics["loss"], 
    ))
    # TODO  Return some info.
    # raise NotImplementedError("Function train_one_epoch is not implemented yet.")
