
"""
Author  Yiqun Chen
Docs    Logger to record information, should not call other custom modules.
"""

import os, sys, logging, time
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter
import copy


class Logger:
    """
    Help user log infomation to file and | or console.
    """
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self.path2logfile = os.path.join(cfg.LOG.DIR, "{}.log".format(cfg.GENERAL.ID))
        logging.basicConfig(filename=self.path2logfile, level=logging.INFO, format='[%(asctime)s] %(message)s')
        self._build()

    def _build(self):
        t = time.gmtime()
        self.path2runs = os.path.join("{}/{}/Mon{}Day{}Hour{}Min{}".format(
            self.cfg.LOG.DIR, 
            "runs", 
            str(t.tm_mon).zfill(2), 
            str(t.tm_mday).zfill(2), 
            str(t.tm_hour).zfill(2), 
            str(t.tm_min).zfill(2), 
        ))
        self.metrics = dict()
        self.writer = SummaryWriter(log_dir=self.path2runs)

    def log_info(self, msg):
        logging.info(msg)
        print(msg)
        # raise NotImplementedError("Method Logger.log_info is not implemented yet.")

    def log_loss(self, tag, loss, global_step=None, walltime=None):
        self.writer.add_scalar(tag, loss, global_step=global_step, walltime=walltime)

    def log_model(self, model, verbose=False):
        inp = torch.randn((2, 6, 256, 256))
        self.writer.add_graph(model=model, input_to_model=[inp], verbose=verbose)

    def log_scalar(self, tag, scalar, global_step=None, walltime=None):
        self.writer.add_scalar(tag, scalar_value=scalar, global_step=global_step, walltime=walltime)

    def close(self):
        self.writer.close()


if __name__ == "__main__":
    pass