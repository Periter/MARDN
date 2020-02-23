
# Multi-resolution Space-attended Residual Dense Network for Single Image Super-Resolution
This repository is for MARDN, and our presentation slide can be download [here](https://drive.google.com/open?id=185_GhtynNQ_rkK6r-HQTcT0w2OXJG3t-).


The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and tested on Ubuntu 18.04/16.04 environment (Python3.6, PyTorch_1.0.1, CUDA9.0, cuDNN7.4) with Nividia RTX 2080/GTX 1080Ti GPUs.

## Contents
1. [Introduction](#introduction)
2. [Results](#results)
3. [Source Code](#source-code)
4. [Acknowledgements](#acknowledgements) 


## Introduction
With the help of deep convolutional neural networks, a vast majority of single image super-resolution (SISR) methods have been developed, and achieved promising performance. However, these methods suffer from over-smoothness in textured regions due to utilizing a single-resolution network to reconstruct both the low-frequency and high-frequency information simultaneously. To overcome this problem, we propose a Multi-resolution space-Attended Residual Dense Network (MARDN) to separate low-frequency and high-frequency information for reconstructing high-quality super-resolved images. Specifically, we start from a low-resolution sub-network, and add low-to-high resolution sub-networks step by step in several stages. These sub-networks with different depth and resolution are utilized to produce feature maps of different frequencies in parallel. For instance, the high-resolution sub-network with fewer stages is applied to local high-frequency textured information extraction, while the low-resolution one with more stages is devoted to generating global low-frequency information. Furthermore, the fusion block with channel-wise sub-network attention is proposed for adaptively fusing the feature maps from different subnetworks instead of applying concatenation and 1 × 1 convolution. A series of ablation investigations and model analyses validate the effectiveness and efficiency of our MARDN. Extensive experiments on benchmark datasets demonstrate the superiority of the proposed MARDN against the state-of-the-art methods.

![MRDA](/imgs/MARDN.png)
Architecture of MARDN. The left part shows the overall structure of our MARDN, which contains four modules: shallow feature extraction, deep feature extraction, upscaling and reconstruction module. While the right part details the deep feature extraction module containing several stages. This module starts from the low-resolution sub-network, then higher resolution sub-networks are added gradually in different stages.

## Results
### Quantitative Results
Quantitative results with the BI degradation model. The best and second best results are highlighted and underlined, respectively.
![PSNR_SSIM_BI](/imgs/QuantityResults.png)

 Quantitative results with the blur-down degradation model. Best and second best results are highlighted and underlined, respectively.
 ![PSNR_SSIM_BI](/imgs/QuantityResults_BD.png)
### Visual Results
Visual comparison for 4× SR with the BI model on the Urban100 and Manga109 datasets. The best results are highlighted.
![Visual_PSNR_SSIM_BI](/imgs/QualityResults.png)
Visual comparison for 3× SR with the BD model on the BSD100 dataset. The best result is highlighted.
![Visual_PSNR_SSIM_BI](/imgs/QualityResults_BD.png)
### More Results
Results on the five benchimark datasets can be downloaded in [Google Drive](https://drive.google.com/open?id=1XbMlpNGv16J_4Rzud0uWgOuQDZ7tqBuH).

## Source code
Source code is available now.
- For training:
    - modify the 'run.sh'
- For testing:
    - download the pretrain model
    - modify the 'run.sh'
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes.
