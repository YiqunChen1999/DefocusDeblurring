
"""
Author:
    Yiqun Chen
Docs:
    Functions to evaluate a model.
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
    cfg, 
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    loss_fn, 
    metrics_logger, 
    phase="valid", 
    logger=None, 
    save=False, 
    *args, 
    **kwargs, 
):
    model.eval()
    # TODO  Prepare to log info.
    log_info = print if logger is None else logger.log_info
    total_loss = []
    # TODO  Read data and evaluate and record info.
    with utils.log_info(msg="{} at epoch: {}".format(phase.upper(), str(epoch).zfill(3)), level="INFO", state=True, logger=logger):
        # log_info("Will{}save results to {}".format(" " if save else " not ", cfg.SAVE.DIR))
        pbar = tqdm(total=len(data_loader), dynamic_ncols=True)
        for idx, data in enumerate(data_loader):
            out, loss = utils.inference_and_cal_loss(model=model, data=data, loss_fn=loss_fn, device=device)
            total_loss.append(loss.detach().cpu().item())

            if save:
                # Save results to directory.
                for i in range(out.shape[0]):
                    save_dir = os.path.join(cfg.SAVE.DIR, phase)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    path2file = os.path.join(save_dir, data["img_idx"][i]+"_g.png")
                    succeed = utils.save_image(out[i].detach().cpu().numpy(), cfg.DATA.MEAN, cfg.DATA.NORM, path2file)
                    if not succeed:
                        log_info("Cannot save image to {}".format(path2file))
            
            metrics_logger.record(phase, epoch, "loss", loss.detach().cpu().item())
            output = out.detach().cpu()
            target = data["target"]
            utils.cal_and_record_metrics(phase, epoch, output, target, metrics_logger, logger=logger)

            pbar.set_description("Epoch: {:<3}, avg loss: {:<5}, cur loss: {:<5}".format(epoch, round(sum(total_loss)/len(total_loss), 5), round(total_loss[-1], 5)))
            pbar.update()
        pbar.close()
    mean_metrics = metrics_logger.mean("valid", epoch)
    log_info("SSIM: {:<5}, PSNR: {:<5}, MAE: {:<5}, Loss: {:<5}".format(
        mean_metrics["SSIM"], mean_metrics["PSNR"], mean_metrics["MAE"], mean_metrics["loss"], 
    ))
    # TODO  Return some info.
    # raise NotImplementedError("Function evaluate is not implemented yet.")

