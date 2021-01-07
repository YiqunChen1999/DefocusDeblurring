
# Defocus Debluring using Dual-pixel Image

 the task of restoring the clean contents from a defocused blurry input image based on a set of prior examples of blurry and clean images. The challenge uses a new dataset and has a single track. Defocus blur is usually undesired and arises in images that are captured with a shallow depth of field. Correcting defocus blur is challenging because the blur is spatially varying and difficult to estimate. One way to think of an effective defocus deblurring is to utilize data available on dual-pixel (DP) sensors found on most modern cameras. DP sensors are used to assist a camera's auto-focus by capturing two sub-aperture views of the scene in a single image shot. **The two sub-aperture images are used to calculate the appropriate lens position to focus on a particular scene region and are discarded afterwards.** DP sub-aperture views provide a good cue for defocus blur present in the scene as they exhibit difference that is correlated to the amount of defocus blur. To address this problem, we proposed to **utilize DP data for effective defocus deblurring**. In this challenge, we provide **a dataset of 600 scenes (2400 images)** where each scene has: 
 
 - (i) an image with defocus blur captured at a large aperture (); 
 - (ii) the two associated DP sub-aperture views; and 
 - (iii) the corresponding all-in-focus image captured with a small aperture (). The target of this challenge is to deblur the input images with the best quantitative results (i.e., PSNR and SSIM) compared to the ground truth. In addition, **efficient solutions** are sought, the inference time of each solution will be measured on **standard desktop CPUs**.

The aim is to obtain a network design / solution capable to produce high quality results with the best fidelity to the reference ground truth.

The top ranked participants will be awarded and invited to follow the CVPR submission guide for workshops to describe their solution and to submit to the associated NTIRE workshop at CVPR 2021.

More details are found on the data section of the competition.
