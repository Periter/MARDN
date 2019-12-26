
# Multi-resolution Space-attended Residual Dense Network for Single Image Super-Resolution
This repository is for MARDN, and our presentation slide can be download [here](https://drive.google.com/open?id=1YRJbslpObygQSd7DniG6HxIV-tNE6B7k).


The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and tested on Ubuntu 18.04/16.04 environment (Python3.6, PyTorch_1.0.1, CUDA9.0, cuDNN7.4) with Nividia RTX 2080Ti/GTX 1080Ti GPUs.

## Contents
1. [Introduction](#introduction)
2. [Results](#results)
3. [Source Code](#source-code)
4. [Acknowledgements](#acknowledgements) 


## Introduction
Single image super-resolution (SISR) has achieved  great success in recent years. A vast majority of SISR methods with high performance were developed using deep convolutional neural networks. However, these methods suffer from over-smoothness in textured regions due to utilizing a single-resolution network to reconstruct both the low-frequency and high-frequency information. To address this problem, we propose a novel SISR method based on Multi-resolution space-Attended Residual Dense Network (MARDN). Specifically, we start from a low-resolution sub-network, and add low-to-high resolution sub-networks step by step in several stages. The sub-networks with different depth and resolution are utilized to produce feature maps of different frequencies in parallel. For instance, the high-resolution sub-network with fewer stages is applied to 
local high-frequency textured information extraction, while the low-resolution one with more stages is devoted to generating global low-frequency information. Furthermore, a fusion block with channel-wise sub-network attention is proposed for adaptively fusing the feature maps from different sub-networks in each stage instead of applying concatenation and 1 x 1 convolution. Extensive experiments on benchmark datasets demonstrate the superiority of the proposed MARDN against the state-of-the-art methods.

![MRDA](/imgs/MARDN.png)
The architecture of our proposed Multi-resolution Space-attended Residual Dense Network for Single Image Super-Resolution (MARDN).

## Results
### Quantitative Results
![PSNR_SSIM_BI](/imgs/QuantityResults.png)


### Visual Results
![Visual_PSNR_SSIM_BI](/imgs/QualityResult.png)

### More Results
Results on the five benchimark datasets can be downloaded in [Google Drive](https://drive.google.com/file/d/133L5zqqWuztqPLLH0WKXajeKWnhwYzEq/view?usp=sharing).

## Source code
Source code will be released upon acceptance of the paper.
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes.
