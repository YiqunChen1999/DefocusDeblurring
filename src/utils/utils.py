
"""
Author:
    Yiqun Chen
Docs:
    Utilities, should not call other custom modules.
"""

import os, sys, copy, functools, time, contextlib
import torch, torchvision
import torch.nn.functional as F
from PIL import Image

@contextlib.contextmanager
def log_info(msg="", level="INFO", state=False, logger=None):
    log = print if logger is None else logger.log_info
    _state = "[{:<8}]".format("RUNNING") if state else ""
    log("[{:<20}] [{:<8}] {} {}".format(time.asctime(), level, _state, msg))
    yield
    if state:
        _state = "[{:<8}]".format("DONE") if state else ""
        log("[{:<20}] [{:<8}] {} {}".format(time.asctime(), level, _state, msg))

def log_info_wrapper(msg, logger=None):
    """
    Decorate factory.
    """
    def func_wraper(func):
        """
        The true decorate.
        """
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            # log = print if logger is None else logger.log_info
            # log("[{:<20}] [{:<8}]".format(time.asctime(), "RUNNING"), msg)
            with log_info(msg=msg, level="INFO", state=True, logger=logger):
                res = func(*args, **kwargs)
            # log("[{:<20}] [{:<8}]".format(time.asctime(), "DONE"), msg)
            return res
        return wrapped_func
    return func_wraper

def inference_and_cal_loss(model, data, loss_fn, device):
    """
    Info:
        Execute inference and calculate loss, sychronize the train and evaluate progress. 
    Args:
        - model (nn.Module): model with train or eval mode setted.
        - inp (dict): organize input data into a dictionary.
        - anno (FIXME): ground truth, used to calculate loss.
        - loss_fn (FIXME): calculate loss.
    Returns:
        - out (FIXME): the output of the model.
        - loss (Tensor): calculated loss.
    """
    l_view, r_view, target = data["l_view"], data["r_view"], data["target"]
    assert len(l_view.shape) == len(r_view.shape) == len(target.shape) == 4, "Incorrect shape."
    inp = torch.cat([l_view, r_view], dim=1)
    inp, target = inp.to(device), target.to(device)
    out = model(inp)
    loss = loss_fn(out, target)
    # raise NotImplementedError("Function inference_and_cal_loss is not implemented yet, \
    #     please rewrite the demo code and delete this error message.")
    return out, loss

def resize(img: torch.Tensor, size: list or tuple, logger=None):
    """
    Info:
        Resize the input image. 
    Args:
        - img (torch.Tensor):
        - size (tuple | int): target size of image.
        - logger (Logger): record running information, if None, direct message to terminal.
    Returns:
        - img (torch.Tensor): image with target size. 
    """
    org_shape = img.shape
    if len(org_shape) == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif len(org_shape) == 3:
        img = img.unsqueeze(0)
    elif len(org_shape) == 4:
        pass
    else:
        raise NotImplementedError("Function to deal with image with shape {} is not implememted yet.".format(org_shape))
    img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
    img = img.reshape(org_shape)
    return img

def set_device(model: torch.nn.Module, gpu_list: list, logger=None):
    with log_info(msg="Set device for model.", level="INFO", state=True, logger=logger):
        if not torch.cuda.is_available():
            with log_info(msg="CUDA is not available, using CPU instead.", level="WARNING", state=False, logger=logger):
                device = torch.device("cpu")
        if len(gpu_list) == 0:
            with log_info(msg="Use CPU.", level="INFO", state=False, logger=logger):
                device = torch.device("cpu")
        elif len(gpu_list) == 1:
            with log_info(msg="Use GPU {}.".format(gpu_list[0]), level="INFO", state=False, logger=logger):
                device = torch.device("cuda:{}".format(gpu_list[0]))
                model = model.to(device)
        elif len(gpu_list) > 1:
            raise NotImplementedError("Multi-GPU mode is not implemented yet.")
    return model, device

def try_make_path_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            return False
    return True

def save_ckpt(path2file, logger=None, **ckpt):
    with log_info(msg="Save checkpoint to {}".format(path2file), level="INFO", state=True, logger=logger):
        torch.save(ckpt, path2file)

def pack_code(cfg, logger=None):
    src_dir = cfg.GENERAL.ROOT
    src_items = [
        "src"
    ]
    des_dir = cfg.LOG.DIR
    with log_info(msg="Pack items {} from ROOT to {}".format(src_items, des_dir), level="INFO", state=True, logger=logger):
        t = time.gmtime()
        for item in src_items:
            path2src = os.path.join(src_dir, item)
            path2des = os.path.join("{}/{}/Mon{}Day{}Hour{}Min{}".format(
                des_dir, 
                "src", 
                str(t.tm_mon).zfill(2), 
                str(t.tm_mday).zfill(2), 
                str(t.tm_hour).zfill(2), 
                str(t.tm_min).zfill(2), 
            ))
            try_make_path_exists(path2des)
            os.system("cp -r {} {}".format(path2src, path2des))
    # raise NotImplementedError("Function pack_code is not implemented yet.")


if __name__ == "__main__":
    log_info(msg="DEBUG MESSAGE", level="DEBUG", state=False, logger=None)
    