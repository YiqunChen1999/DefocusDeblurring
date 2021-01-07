
"""
Author:
    Yiqun Chen
Docs:
    Configurations, should not call other custom modules.
"""

import os, sys, copy, argparse
from attribdict import AttribDict as Dict

configs = Dict()
cfg = configs

parser = argparse.ArgumentParser()
parser.add_argument("id", type=str)
parser.add_argument("strict_id", default="true", choices=["true", "false"], type=str)
parser.add_argument("batch_size", type=int)
parser.add_argument("train", default="true", choices=["true", "false"], type=str)
parser.add_argument("eval", default="true", choices=["true", "false"], type=str)
parser.add_argument("test", default="false", choices=["true", "false"], type=str)
parser.add_argument("resume", default="false", choices=["true", "false"], type=str)
parser.add_argument("gpu", type=str)
args = parser.parse_args()

# ================================ 
# GENERAL
# ================================ 
cfg.GENERAL.ROOT                                =   os.path.join(os.getcwd(), "..", "..")
cfg.GENERAL.ID                                  =   "{}".format(args.id)
cfg.GENERAL.STRICT_ID                           =   True if args.strict_id == "true" else False
cfg.GENERAL.BATCH_SIZE                          =   args.batch_size
cfg.GENERAL.TRAIN                               =   True if args.train == "true" else False
cfg.GENERAL.EVAL                                =   True if args.eval == "true" else False
cfg.GENERAL.TEST                                =   True if args.test == "true" else False
cfg.GENERAL.RESUME                              =   True if args.resume == "true" else False
cfg.GENERAL.GPU                                 =   eval(args.gpu)

# ================================ 
# MODEL
# ================================ 
cfg.MODEL.ARCH                                  =   None # TODO
cfg.MODEL.ENCODER                               =   "DPDEncoder"
cfg.MODEL.DECODER                               =   "DPDDncoder"
cfg.MODEL.CKPT_DIR                              =   os.path.join(cfg.GENERAL.ROOT, "checkpoints", cfg.GENERAL.ID)
cfg.MODEL.PATH2CKPT                             =   os.path.join(cfg.MODEL.CKPT_DIR, cfg.GENERAL.ID + ".pth")

# ================================ 
# DATA
# ================================ 
cfg.DATA.DIR                                    =   ""
cfg.DATA.NUMWORKERS                             =   args.batch_size
cfg.DATA.DATASET                                =   "DualPixelNTIRE2021"

# ================================ 
# OPTIMIZER
# ================================ 
cfg.OPTIMIZER.OPTIMIZER                         =   "Adam"
cfg.OPTIMIZER.LR                                =   2e-5

# ================================ 
# SCHEDULER
# ================================ 
cfg.SCHEDULER.SCHEDULER                         =   "StepLRScheduler"

# ================================ 
# SCHEDULER
# ================================ 
cfg.TRAIN.MAX_EPOCH                             =   200

# ================================ 
# LOSS_FN
# ================================ 
cfg.LOSS_FN.LOSS_FN                             =   "MSELoss"

# ================================ 
# LOG
# ================================ 
cfg.SAVE.DIR                                    =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "results", cfg.GENERAL.ID))
cfg.SAVE.SAVE                                   =   False

# ================================ 
# LOG
# ================================ 
cfg.LOG.DIR                                     =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "logs", cfg.GENERAL.ID))

if cfg.GENERAL.STRICT_ID:
    assert not os.path.exists(cfg.LOG.DIR), "Cannot use same ID in strict mode."
    
_paths = [
    cfg.LOG.DIR, 
    cfg.MODEL.CKPT_DIR, 
    cfg.SAVE.DIR, 
]

for _path in _paths:
    if not os.path.exists(_path):
        os.makedirs(_path)

raise NotImplementedError("Please set your configurations and remove this error message.")