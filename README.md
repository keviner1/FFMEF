Official PyTorch implementation of our CVPRW2023 paper: Efficient Multi-exposure Image Fusion via Filter-dominated Fusion and Gradient-driven Unsupervised Learning.
[Paper link](https://openaccess.thecvf.com/content/CVPR2023W/MIPI/papers/Zheng_Efficient_Multi-Exposure_Image_Fusion_via_Filter-Dominated_Fusion_and_Gradient-Driven_Unsupervised_CVPRW_2023_paper.pdf)

-------------------------------------------------
**Frameworks**

*FFMEF  &  GIFloss*

<img src="https://github.com/keviner1/imgs/blob/main/FFMEF-model.png?raw=true" width="400px"> <img src="https://github.com/keviner1/imgs/blob/main/FFMEF-loss.png?raw=true" width="263px">

-------------------------------------------------
**Results**

*multi-exposure fusion*
![show](https://github.com/keviner1/imgs/blob/main/FFMEF-comp.png?raw=true)

*multi-focus fusion  &  visible-infrared fusion*
![show](https://github.com/keviner1/imgs/blob/main/FFMEF-comp2.png?raw=true)

-------------------------------------------------
**We provide a simple training and testing process as follows:**

-------------------------------------------------
**Dependencies**
* Python 3.8
* PyTorch 1.10.0+cu113

-------------------------------------------------
**Train**

The datasets samples are placed in *images\dataset* (including MEFB[1], MFIF[2], VIFB[3], and SICE[4]).

> Multi-Exposure Image Fusion (MEF)

python train.py --config 1

> Multi-Focus Image Fusion (MFF)

python train.py --config 2

> Visible-Infrared Image Fusion (VIF)

python train.py --config 3

Then, the checkpoints and log file are saved in *output*.

-------------------------------------------------
**Test**

The pretrained models are placed in *ckp*.

> MEF

python test.py --config 1 --ckp mef.pth

> MFF

python test.py --config 2 --ckp mff.pth

> VIF

python test.py --config 3 --ckp vif.pth

-------------------------------------------------
Finally, the fused results can be found in *images\fused*.

-------------------------------------------------
**Citation**

```
@inproceedings{zheng2023ffmef,
  title={Efficient Multi-exposure Image Fusion via Filter-dominated Fusion and Gradient-driven Unsupervised Learning},
  author={Zheng, Kaiwen and Huang, Jie and Yu, Hu and Zhao, Feng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2023}
}
```
-------------------------------------------------
**Reference**

[1] Zhang X. Benchmarking and comparing multi-exposure image fusion algorithms[J]. Information Fusion, 2021, 74: 111-131.

[2] Zhang X. Deep learning-based multi-focus image fusion: A survey and a comparative study[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.

[3] Zhang X, Ye P, Xiao G. VIFB: A visible and infrared image fusion benchmark[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2020: 104-105.

[4] Cai J, Gu S, Zhang L. Learning a deep single image contrast enhancer from multi-exposure images[J]. IEEE Transactions on Image Processing, 2018, 27(4): 2049-2062.


