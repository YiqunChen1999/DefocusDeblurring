
"""
Author:
    Yiqun Chen
Docs:
    Metrics.
"""

import copy, skimage, math, sklearn
import numpy as np
from sklearn import metrics
from skimage import metrics

def cal_mae(y_true, y_pred, *args, **kwargs):
    if y_true.shape[0] == 3:
        y_true = np.transpose(y_true, (1, 2, 0))
        y_pred = np.transpose(y_pred, (1, 2, 0))
    mae_0 = sklearn.metrics.mean_absolute_error(y_true[:,:,0], y_pred[:,:,0])
    mae_1 = sklearn.metrics.mean_absolute_error(y_true[:,:,1], y_pred[:,:,1])
    mae_2 = sklearn.metrics.mean_absolute_error(y_true[:,:,2], y_pred[:,:,2])
    return np.mean([mae_0,mae_1,mae_2])

def cal_psnr(image_true, image_test, data_range=None, *args, **kwargs):
    psnr = skimage.metrics.peak_signal_noise_ratio(image_true, image_test, data_range=data_range)
    return psnr
    
def cal_ssim(
    im1, 
    im2, 
    data_range=None, 
    multichannel=True, 
    *args, 
    **kwargs
):
    if im1.shape[0] == 3:
        im1 = np.transpose(im1, (1, 2, 0))
        im2 = np.transpose(im2, (1, 2, 0))
    ssim = skimage.metrics.structural_similarity(im1, im2, data_range=data_range, multichannel=multichannel)
    return ssim


class Metrics:
    def __init__(self):
        self.metrics = {}

    def record(self, phase, epoch, item, value):
        if phase not in self.metrics.keys():
            self.metrics[phase] = {}
        if epoch not in self.metrics[phase].keys():
            self.metrics[phase][epoch] = {}
        if item not in self.metrics[phase][epoch].keys():
            self.metrics[phase][epoch][item] = []
        self.metrics[phase][epoch][item].append(value)

    def get_metrics(self, phase=None, epoch=None, item=None):
        metrics = copy.deepcopy(self.metrics)
        if phase is not None:
            metrics = {phase: metrics[phase]}
        if epoch is not None:
            for _p in metrics.keys():
                metrics[_p] = {epoch: metrics[_p][epoch]}
        if item is not None:
            for _p in metrics.keys():
                for _e in metrics[_p].keys():
                    metrics[_p][_e] = {item: metrics[_p][_e][item]}
        return metrics

    def mean(self, phase, epoch, item=None):
        mean_metrics = {}
        metrics = self.get_metrics(phase=phase, epoch=epoch, item=item)
        metrics = metrics[phase][epoch]
        for key, value in metrics.items():
            mean_metrics[key] = np.mean(np.array(value))
        return mean_metrics

    def cal_metrics(self, phase, epoch, *args, **kwargs):
        mae = cal_mae(*args, **kwargs)
        ssim = cal_ssim(*args, **kwargs)
        psnr = cal_psnr(*args, **kwargs)
        self.record(phase, epoch, "MAE", mae)
        self.record(phase, epoch, "SSIM", ssim)
        self.record(phase, epoch, "PSNR", psnr)
        return mae, ssim, psnr


if __name__ == "__main__":
    # metrics_logger = Metrics()
    # for phase in ["train", "valid", "test"]:
    #     for epoch in range(20):
    #         for item in ["mse", "psnr", "ssim"]:
    #             metrics_logger.record(phase, epoch, item, np.random.randn(1))
    # metrics_logger.get_metrics()
    # print(metrics_logger.mean("train", 0, item=None))
    import cv2
    path2img1 = "/mnt/g/Datasets/DualPixelNTIRE2021/train/target/00000.png"
    path2img2 = "/mnt/g/Datasets/DualPixelNTIRE2021/train/l_view/00000.png"
    img1 = cv2.imread(path2img1, -1) / (2**16-1)
    img2 = cv2.imread(path2img2, -1) / (2**16-1)
    print(cal_mae(img1, img2))
    print(cal_ssim(img1, img2, data_range=1, multichannel=True))
    print(cal_psnr(img1, img2, data_range=1))

    # 0.031292318003256575 25.087839612030397 0.8068734330381345
    # 0.031292318003256575 25.087839612030397 0.8068734330381345
