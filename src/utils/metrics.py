
"""
Author:
    Yiqun Chen
Docs:
    Metrics.
"""

import skimage, math, sklearn
import numpy as np

def cal_mae(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    mae_0 = sklearn.metrics.mean_absolute_error(img1[:,:,0], img2[:,:,0], sample_weight=sample_weight, multioutput=multioutput)
    mae_1 = sklearn.metrics.mean_absolute_error(img1[:,:,1], img2[:,:,1], sample_weight=sample_weight, multioutput=multioutput)
    mae_2 = sklearn.metrics.mean_absolute_error(img1[:,:,2], img2[:,:,2], sample_weight=sample_weight, multioutput=multioutput)
    return np.mean([mae_0,mae_1,mae_2])

def cal_psnr(image_true, image_test, data_range=None):
    psnr = skimage.metrics.peak_signal_noise_ratio(image_true, image_test, data_range=None)
    return psnr
    
def cal_ssim(
    im1, 
    im2, 
    win_size=None, 
    gradient=False, 
    data_range=None, 
    multichannel=False, 
    gaussian_weights=False, 
    full=False, 
    **kwargs
):
    ssim = skimage.metrics.structural_similarity(img1, img2, win_size=win_size, gradient=gradient, data_range=PIXEL_MAX, multichannel=True, gaussian_weights=False, full=False)
    return ssim

