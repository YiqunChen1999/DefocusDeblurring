
"""
Author:
    Yiqun Chen
Docs:
    Main functition to run program.
"""

import sys, os, copy
import torch, torchvision

from configs.configs import cfg
from utils import utils, loss_fn_helper, lr_scheduler_helper, optimizer_helper
from utils.logger import Logger
from models import model_builder
from data import data_loader
from train import train_one_epoch
from evaluate import evaluate
from utils.metrics import Metrics

def main():
    # raise NotImplementedError("Function main is not implemented yet, please finish your code and \
    #     remove this error message.")
    # TODO Read configuration.
    # TODO Set logger to record information.
    logger = Logger(cfg)
    logger.log_info(cfg)
    metrics_logger = Metrics()
    utils.pack_code(cfg, logger=logger)
    # TODO Build model.
    model = model_builder.build_model(cfg=cfg, logger=logger)
    # logger.log_model(model)
    # TODO Read checkpoint.
    ckpt = torch.load(cfg.MODEL.PATH2CKPT) if cfg.GENERAL.RESUME else {}
    # TODO Load pre-trained model.
    model = model.load_state_dict(ckpt["model"]) if cfg.GENERAL.RESUME else model
    resume_epoch = ckpt["epoch"] if cfg.GENERAL.RESUME in ckpt.keys() else 0
    optimizer = ckpt["optimizer"] if cfg.GENERAL.RESUME else optimizer_helper.build_optimizer(cfg=cfg, model=model)
    lr_scheduler = ckpt["lr_scheduler"] if cfg.GENERAL.RESUME else lr_scheduler_helper.build_scheduler(cfg=cfg, optimizer=optimizer)
    loss_fn = ckpt["loss_fn"] if cfg.GENERAL.RESUME else loss_fn_helper.build_loss_fn(cfg=cfg)
    # TODO Set device.
    model, device = utils.set_device(model, cfg.GENERAL.GPU)
    # TODO Prepare dataset.
    train_data_loader = data_loader.build_data_loader(cfg, cfg.DATA.DATASET, "train")
    try:
        valid_data_loader = data_loader.build_data_loader(cfg, cfg.DATA.DATASET, "valid")
    except:
        with utils.log_info(msg="Cannot build valid dataset.", level="ERROR", state=False):
            pass

    # TODO Train, evaluate model and save checkpoint.
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        if resume_epoch > epoch:
            continue
        train_one_epoch(
            epoch=epoch, 
            model=model, 
            data_loader=train_data_loader, 
            device=device, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            metrics_logger=metrics_logger, 
            logger=logger, 
        )
        utils.save_ckpt(
            path2file=os.path.join(cfg.MODEL.CKPT_DIR, cfg.GENERAL.ID + "_" + str(epoch).zfill(3) + ".pth"), 
            logger=logger, 
            model=model.state_dict(), 
            epoch=epoch, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            loss_fn=loss_fn, 
        )
        try:
            evaluate(
                epoch=epoch, 
                model=model, 
                data_loader=valid_data_loader, 
                device=device, 
                loss_fn=loss_fn, 
                metrics_logger=metrics_logger, 
                phase="valid", 
                logger=logger,
                save=cfg.SAVE.SAVE,  
            )
        except:
            with utils.log_info(msg="Cannot evaluate model.", level="ERROR", state=False):
                pass
    return None


if __name__ == "__main__":
    main()


