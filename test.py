#!/usr/local/bin/python
import sys
# sys.path.insert(1, '/data/pylib')
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import datetime
import random
import numpy as np
import argparse
import model.FFMEF as FFMEF
from config import config_1
from config import config_2
from config import config_3
from torch.utils.data import DataLoader, DistributedSampler
import myUtils
from skimage.metrics import structural_similarity
from pathlib import Path
import logging
from PIL import Image
from thop import profile
import torchvision.utils as vutils
from einops import rearrange
import math
import time
import datetime
import cv2

def save(img,path,suf,flag,mode,ckp):
    cnt = path
    #img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0).squeeze()

    img = img.permute([0, 2, 3, 1]).cpu().detach().numpy()
    img = np.squeeze(img)
    
    img = np.uint8(img)
    img = img.astype(np.uint8)
    save_img = Image.fromarray(img, mode)#
    if not os.path.exists(f"images/fused/{ckp}"):
        os.makedirs(f"images/fused/{ckp}")
    save_img.save(f"images/fused/{ckp}/{cnt}_{flag}.{suf}")

def run(args, ckp):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ---------------------------------main_model--------------------------------
    if args.MODEL.choice == "FFMEF":
        model = FFMEF.FFMEF(channels=args.MODEL.channels)

    checkpoint = torch.load(f"ckp/{ckp}")
    net = checkpoint['model']
    net = {key.replace("module.", ""): val for key, val in net.items()}
    model.load_state_dict(net,strict=False)
    model.to(device)
    #--------------------------------
    test_RGB = torch.randn(2,3,256,256).cuda()
    test_Y = torch.randn(2,1,256,256).cuda()
    model.eval()
    flops, params = profile(model, (test_RGB, test_Y, test_RGB, test_Y))
    print(f'flops: {(flops/1024/1024/1024):.4f}G;  params: {(params/1024/1024):.4f}M')
    #--------------------------------
    dataset_test = myUtils.build_dataset(args, mode='test')
    n = len(dataset_test)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_val = DataLoader(dataset_test, batch_size=1,sampler=sampler_test, num_workers=1,pin_memory=True)

    start = time.time()
    with torch.no_grad():
        for _, samples in enumerate(dataloader_val):
            img0_RGB = samples[0]['img0_RGB'].cuda()
            img0_gra = samples[0]['img0_gra'].cuda()
            img1_RGB = samples[0]['img1_RGB'].cuda()
            img1_gra = samples[0]['img1_gra'].cuda()
            img1_Y = samples[0]['img1_Y'].cuda()
            img1_Cb = samples[0]['img1_Cb'].cuda()
            img1_Cr = samples[0]['img1_Cr'].cuda()
            img0_Y = samples[0]['img0_Y'].cuda()
            img0_Cb = samples[0]['img0_Cb'].cuda()
            img0_Cr = samples[0]['img0_Cr'].cuda()
            
            #===========================================================================
            fus_Y = model(img0_RGB = None, img0_Y = img0_Y, img1_RGB = None, img1_Y = img1_Y)
            fus_Y = fus_Y.clamp(0,1)

            if args.DATA.task == "MEF" or args.DATA.task == "MFF":
                fus_Cb = myUtils.colorCombine_torch([img0_Cb*255, img1_Cb*255]) #[B C H W], 255
                fus_Cr = myUtils.colorCombine_torch([img0_Cr*255, img1_Cr*255])
                fus_RGB = myUtils.YCbCr2RGB_torch(fus_Y*255,fus_Cb,fus_Cr)
            elif args.DATA.task == "VIF":
                fus_RGB = myUtils.YCbCr2RGB_torch(fus_Y*255,img1_Cb*255,img1_Cr*255)      #B C H W, 255

            save(img0_RGB*255.0,samples[1][0],samples[2][0],"I0RGB","RGB",ckp)
            save(img1_RGB*255.0,samples[1][0],samples[2][0],"I1RGB","RGB",ckp)
            save(fus_RGB,samples[1][0],samples[2][0],"FFMEF","RGB",ckp)
            
    end = time.time()
    gap = round(end * 1000) - round(start * 1000)
    print(f"time_avg: {gap/n:.4f}ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default = -1)
    parser.add_argument("--config", default = 1, type=int)
    parser.add_argument("--ckp")
    args = parser.parse_args()
    if args.config == 1:
        run(config_1.get_config(), args.ckp)
    if args.config == 2:
        run(config_2.get_config(), args.ckp)
    if args.config == 3:
        run(config_3.get_config(), args.ckp)



