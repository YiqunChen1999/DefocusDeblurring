
"""
Author  Yiqun Chen
Docs    Test modules not model.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
from utils import utils

@utils.log_info_wrapper(msg="Another start info", logger=None)
def test():
    print("Hello World!")


if __name__ == "__main__":
    with utils.log_info(msg="Start test", level="INFO", state=True, logger=None):
        test()