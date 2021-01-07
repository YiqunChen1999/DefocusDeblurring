
"""
Author  Yiqun Chen
Docs    Test modules not model.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
from torch import nn
import torch.nn.functional as F
from utils import utils
from configs.configs import cfg
from models import model_builder

@utils.log_info_wrapper(msg="Another start info", logger=None)
def test():
    print("Hello World!")

def test_model():
    # from configs.configs import cfg
    model = model_builder.build_model(cfg=cfg, logger=None)
    inp = torch.randn((2, 6, 512, 512))
    out = model(inp)
    print(model)


if __name__ == "__main__":
    with utils.log_info(msg="Start test", level="INFO", state=True, logger=None):
        # test()
        test_model()