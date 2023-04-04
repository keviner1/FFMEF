#!/usr/local/bin/python
import sys
import os
import time
import argparse
import time
import datetime
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import sys
src = str(os.path.split(os.path.realpath(__file__))[0]).replace("\\","/")
from PIL import Image
from PIL import ImageDraw
import glob
import math
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
# from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import cv2
import logging
from einops import rearrange
import cv2
import random
from torch.cuda.amp import autocast as autocast, GradScaler
# from mef_ssim import MEF_SSIM_Loss
# from PMGI import PMGI_Loss
from GIF import GIF_Loss


class My_Dataset(Dataset):
    def __init__(self,args,mode='train'):
        self.crop_resize = args.DATA.crop_resize
        self.mode = mode

        if mode == "train":
            self.src = args.SERVER.TRAIN_DATA
            self.files = os.listdir(self.src)
            self.files = sorted(self.files)[:args.SERVER.train_len]
        elif mode == "val":
            self.src = args.SERVER.VAL_DATA
            self.files = os.listdir(self.src)
            self.files = sorted(self.files)[:args.SERVER.val_len]
        elif mode == "test":
            self.src = args.SERVER.TEST_DATA
            self.files = os.listdir(self.src)
            self.files = sorted(self.files)[:args.SERVER.test_len]

    def __len__(self):
        return len(self.files)
    
    def random_prob(self, h, w):
        hflip = random.random()
        vflip = random.random()
        h_r = random.randint(5, 10)/10   #half crop
        w_r = random.randint(5, 10)/10
        new_h = math.floor(h_r * h)
        new_w = math.floor(w_r * w)
        h_range = h - new_h
        w_range = w - new_w
        y1 = random.randint(0, h_range)
        x1 = random.randint(0, w_range)
        cropIndex = [x1, y1, x1+new_h, y1+new_w]
        return hflip, vflip, cropIndex

    def img_aug(self, img, hflip, vflip, cropIndex, crop_resize):
        if hflip >= 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if vflip >= 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.crop(cropIndex)
        img = img.resize(crop_resize)
        return img

    def ImgRead(self, file, hflip, vflip, cropIndex):
        if self.mode == "train":
            img = cv2.imread(glob.glob(file+".*")[0],-1)
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            img =  self.img_aug(img, hflip, vflip, cropIndex, crop_resize = self.crop_resize)
        elif self.mode == "val":
            img = cv2.imread(glob.glob(file+".*")[0],-1)
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            img = img.resize((320,320))
        elif self.mode == "test":
            img = cv2.imread(glob.glob(file+".*")[0],-1)
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            # img = img.resize((320,320))
        img = np.array(img, dtype=np.uint8).astype(np.float32)
        return img

    def __getitem__(self, index):
        name = self.files[index]
        path = self.src + "/" + name
        # print(path)
        #=====================prepare for aug==================================
        temp = glob.glob(path + f'/{name}_A.*')[0]
        img = cv2.imread(temp)
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        suffix = temp.split(".")[-1]
        (w,h) = img.size
        hflip, vflip, cropIndex = self.random_prob(h = h, w = w)
        #=====================reading and augmentation==========================
        img0 = self.ImgRead(path + f'/{name}_A', hflip, vflip, cropIndex)
        img1 = self.ImgRead(path + f'/{name}_B', hflip, vflip, cropIndex)

        #=====================YCBCR============================================
        img0_Y, img0_Cb, img0_Cr = RGB2YCbCr_np(img0)
        img1_Y, img1_Cb, img1_Cr = RGB2YCbCr_np(img1)

        #=====================gradient============================================
        img0_gra = np.array(getGradient(img0_Y), dtype=np.uint8).astype(np.float32)
        img1_gra = np.array(getGradient(img1_Y), dtype=np.uint8).astype(np.float32)

        img0_RGB = ToTensor()(img0) / 255.0
        img0_gra = ToTensor()(img0_gra) / 255.0
        img1_RGB = ToTensor()(img1) / 255.0
        img1_gra = ToTensor()(img1_gra) / 255.0
        img1_Y = ToTensor()(img1_Y) / 255.0
        img1_Cb = ToTensor()(img1_Cb) / 255.0
        img1_Cr = ToTensor()(img1_Cr) / 255.0
        img0_Y = ToTensor()(img0_Y) / 255.0
        img0_Cb = ToTensor()(img0_Cb) / 255.0
        img0_Cr = ToTensor()(img0_Cr) / 255.0
        samples = {
            "img0_RGB": img0_RGB,
            "img0_gra": img0_gra,
            "img1_RGB": img1_RGB,
            "img1_gra": img1_gra,
            "img1_Y": img1_Y,
            "img1_Cb": img1_Cb,
            "img1_Cr": img1_Cr,
            "img0_Y": img0_Y,
            "img0_Cb": img0_Cb,
            "img0_Cr": img0_Cr,
        }
        return samples, name, suffix

def build_dataset(args, mode):
    return My_Dataset(args,mode)


#------------------------------------------------------------------------------------------------------
class L1Loss_W(nn.Module):
    def __init__(self):
        super(L1Loss_W, self).__init__()
    
    def forward(self, y1, y2, Weight):
        dis = torch.abs(y1-y2)
        dis = dis * Weight
        return torch.mean(dis)

class L2Loss_W(nn.Module):
    def __init__(self):
        super(L2Loss_W, self).__init__()
    
    def forward(self, y1, y2, Weight):
        dis = torch.pow((y1-y2),2)
        dis = dis * Weight
        return torch.mean(dis)

def save(img,temp_path,cnt,flag,suf,mode):
    img = img[0].unsqueeze(0)
    img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0).squeeze()
    img = np.uint8(img)
    img = img.astype(np.uint8)
    save_img = Image.fromarray(img, mode)#
    save_img.save(f"{temp_path}/{cnt}_{flag}.{suf}")

#-------------------------------------Trian/validation one epoch ------------------------------------------
def train_one_epoch(args, model, dataloader_train, optimizer, lr_scheduler, epoch, logger, scaler):
    model.train()
    l1_loss = torch.nn.L1Loss().cuda()
    l1_loss.eval()
    l1_loss_W = L1Loss_W()
    l1_loss_W.eval()
    l2_loss_W = L2Loss_W()
    l2_loss_W.eval()
    l2_loss = torch.nn.MSELoss().cuda()
    l2_loss.eval()
    # mefssim_Loss = MEF_SSIM_Loss() 
    # mefssim_Loss.eval()
    gra_Loss = GIF_Loss() 
    gra_Loss.eval()
    # pmgi_Loss = PMGI_Loss() 
    # pmgi_Loss.eval()
    optimizer.zero_grad()
    num_steps = len(dataloader_train)
    start = time.time()
    totalloss = 0
    for _, samples in enumerate(dataloader_train):
        img0_RGB = samples[0]['img0_RGB'].cuda()
        img0_Y = samples[0]['img0_Y'].cuda()
        img0_Cb = samples[0]['img0_Cb'].cuda()
        img0_Cr = samples[0]['img0_Cr'].cuda()
        img0_gra = samples[0]['img0_gra'].cuda()
        img1_RGB = samples[0]['img1_RGB'].cuda()
        img1_Y = samples[0]['img1_Y'].cuda()
        img1_Cb = samples[0]['img1_Cb'].cuda()
        img1_Cr = samples[0]['img1_Cr'].cuda()
        img1_gra = samples[0]['img1_gra'].cuda()
        #===========================================================================
        fus_Y = model(img0_RGB = img0_RGB, img0_Y = img0_Y, img1_RGB = img1_RGB, img1_Y = img1_Y)
        fus_Y = fus_Y.clamp(0,1)
        if args.TRAIN.loss == "GIF":
            loss_g = gra_Loss(img0_Y, img0_gra, img1_Y, img1_gra, img0_Cb, img0_Cr, img1_Cb, img1_Cr, fus_Y, "")
            loss = loss_g
        # if args.TRAIN.loss == "MEF-SSIM":
        #     loss_mef = mefssim_Loss(img0_Y, img1_Y, fus_Y)
        #     loss = loss_mef
        # if args.TRAIN.loss == "PMGI":
        #     loss_pmgi = pmgi_Loss(img0_Y, img1_Y, fus_Y)
        #     loss = loss_pmgi
        #----------------------------------------------------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(epoch)
        cur_lr=optimizer.param_groups[-1]['lr']
        totalloss = totalloss + loss.item()
        if _ % args.TRAIN.PRINT_FREQ == 0 and (args.distributed == False or torch.distributed.get_rank() == 0):
            logger.info(f"||train -- epoch:{epoch} / {args.TRAIN.EPOCHS} --step:{_+1} / {num_steps} --lr:{cur_lr:.7f}-----curloss: {loss.item():.5f}")
    loss_avg = totalloss / num_steps
    epoch_time = time.time() - start 
    if args.distributed == False or torch.distributed.get_rank() == 0:
        Logger_msg = f":-------------------------Train: Epoch:{epoch:04}, timespend:{datetime.timedelta(seconds=int(epoch_time))}, avg_loss:{loss_avg:.5f}"
        logger.info(Logger_msg)
        

@torch.no_grad()
def validate(args, model, dataloader_val, epoch, logger, dataset_val_len):
    model.eval()
    l1_loss = torch.nn.L1Loss().cuda()
    l1_loss.eval()
    l1_loss_W = L1Loss_W()
    l1_loss_W.eval()
    l2_loss_W = L2Loss_W()
    l2_loss_W.eval()
    l2_loss = torch.nn.MSELoss().cuda()
    l2_loss.eval()
    l1_loss_W = L1Loss_W()
    l1_loss_W.eval()
    # mefssim_Loss = MEF_SSIM_Loss() 
    # mefssim_Loss.eval()
    gra_Loss = GIF_Loss() 
    gra_Loss.eval()
    # pmgi_Loss = PMGI_Loss() 
    # pmgi_Loss.eval()
    num_steps = len(dataloader_val)
    start = time.time()
    totalloss = 0
    for _, samples in enumerate(dataloader_val):
        if _ % args.TRAIN.PRINT_FREQ == 0 and (args.distributed == False or torch.distributed.get_rank() == 0):
            logger.info(f"|| Val -- epoch:{epoch} / {args.TRAIN.EPOCHS} ------step:{_+1} / {num_steps}")
        img0_RGB = samples[0]['img0_RGB'].cuda()
        img0_Y = samples[0]['img0_Y'].cuda()
        img0_Cb = samples[0]['img0_Cb'].cuda()
        img0_Cr = samples[0]['img0_Cr'].cuda()
        img0_gra = samples[0]['img0_gra'].cuda()
        img1_RGB = samples[0]['img1_RGB'].cuda()
        img1_Y = samples[0]['img1_Y'].cuda()
        img1_Cb = samples[0]['img1_Cb'].cuda()
        img1_Cr = samples[0]['img1_Cr'].cuda()
        img1_gra = samples[0]['img1_gra'].cuda()
        #===========================================================================
        fus_Y = model(img0_RGB = img0_RGB, img0_Y = img0_Y, img1_RGB = img1_RGB, img1_Y = img1_Y)
        fus_Y = fus_Y.clamp(0,1)

        if args.TRAIN.loss == "GIF":
            loss_g = gra_Loss(img0_Y, img0_gra, img1_Y, img1_gra, img0_Cb, img0_Cr, img1_Cb, img1_Cr, fus_Y, samples[1][0])
            loss = loss_g
        # if args.TRAIN.loss == "MEF-SSIM":
        #     loss_mef = mefssim_Loss(img0_Y, img1_Y, fus_Y)
        #     loss = loss_mef
        # if args.TRAIN.loss == "PMGI":
        #     loss_pmgi = pmgi_Loss(img0_Y, img1_Y, fus_Y)
        #     loss = loss_pmgi

        fus_RGB = YCbCr2RGB_torch(fus_Y*255,img1_Cb*255,img1_Cr*255)      #B C H W, 255
        
        temp_path = os.path.join(args.SERVER.OUTPUT+f"/temp")
        save(fus_RGB, temp_path, samples[1][0], "SFNet", samples[2][0], "RGB")
        save(img0_RGB*255, temp_path, samples[1][0], "img0_RGB", samples[2][0], "RGB")
        save(img1_RGB*255, temp_path, samples[1][0], "img1_RGB", samples[2][0], "RGB")
        totalloss = totalloss + loss.item()
        #==========================================================================
    loss_avg = totalloss / num_steps
    epoch_time = time.time() - start 
    #
    if args.distributed == False or torch.distributed.get_rank() == 0:
        Logger_msg = f":Validation: Epoch:{epoch:04}, timespend:{datetime.timedelta(seconds=int(epoch_time))}, loss_avg:{loss_avg:.4f}"
        logger.info(Logger_msg)
    return loss_avg

#-----------------------------------YCbCr--transform------------------------------------------
def colorCombine_np(seq_C): # [H W C]
	a = np.zeros(seq_C[0].shape)
	b = np.zeros(seq_C[0].shape)
	for C in seq_C:
		a = a + C*np.abs(C - 128)
		b = b + np.abs(C - 128)
	new = np.nan_to_num(a/b, copy=True, nan=128, posinf=None, neginf=None)
	return new

def RGB2YCbCr_np(img): #H W C
	R = img[:,:,0:1]
	G = img[:,:,1:2]
	B = img[:,:,2:3]
	Y = 0.256789 * R + 0.504129 * G + 0.097906 * B + 16
	Cb = -0.148223 * R - 0.290992 * G + 0.439215 * B + 128
	Cr = 0.439215 * R - 0.367789 * G - 0.071426 * B + 128
	return Y,Cb,Cr

def YCbCr2RGB_np(Y,Cb,Cr): #H W C
	R = 1.164383 * (Y-16) + 1.596027 * (Cr-128)
	G = 1.164383 * (Y-16) - 0.391762 * (Cb-128)- 0.812969 * (Cr-128)
	B = 1.164383 * (Y-16) + 2.017230 * (Cb-128)
	R = np.clip(R, 0, 255)
	G = np.clip(G, 0, 255)
	B = np.clip(B, 0, 255)
	new = np.concatenate((R,G,B),axis = 2)
	return new

def colorCombine_torch(seq_C): #[B C H W]
	a = torch.zeros(seq_C[0].shape).cuda()
	b = torch.zeros(seq_C[0].shape).cuda()
	for C in seq_C:
		a = a + C*torch.abs(C - 128.0)
		b = b + torch.abs(C - 128.0)
	new = a/b
	new[torch.isnan(new)] = 128.0
	return new

def RGB2YCbCr_torch(img): #B C H W
	R = img[:,0:1,:,:]
	G = img[:,1:2,:,:]
	B = img[:,2:3,:,:]
	Y = 0.256789 * R + 0.504129 * G + 0.097906 * B + 16
	Cb = -0.148223 * R - 0.290992 * G + 0.439215 * B + 128
	Cr = 0.439215 * R - 0.367789 * G - 0.071426 * B + 128
	return Y,Cb,Cr

def YCbCr2RGB_torch(Y,Cb,Cr): #B C H W
	R = 1.164383 * (Y-16.0) + 1.596027 * (Cr-128.0)
	G = 1.164383 * (Y-16.0) - 0.391762 * (Cb-128.0)- 0.812969 * (Cr-128.0)
	B = 1.164383 * (Y-16.0) + 2.017230 * (Cb-128.0)
	R = torch.clip(R, 0, 255)
	G = torch.clip(G, 0, 255)
	B = torch.clip(B, 0, 255)
	new = torch.cat([R,G,B],1)
	return new

def getGradient(img):
    scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
    scharry = cv2.Scharr(img,cv2.CV_64F,0,1)
    scharrx = np.abs(scharrx)
    scharry = np.abs(scharry)
    scharrx = np.clip(scharrx, 0, 255)
    scharry = np.clip(scharry, 0, 255)
    G =  scharrx*0.5 + scharry*0.5  
    # scharrx = cv2.convertScaleAbs(scharrx)   
    # scharry = cv2.convertScaleAbs(scharry) 
    # G =  cv2.addWeighted(scharrx,0.5,scharry,0.5,0)  
    return G

def MaxMinNorm(x, max, min):
    return (x - min) / (max - min)

#-------------------------------------metrics ------------------------------------------
def cal_psnr(imag1, imag2):     # B C H W tensor [0 ~ 1]
    # imag1 = imag1 * 255.0
    # imag2 = imag2 * 255.0
    sum = 0
    for b in range(imag1.shape[0]):
        im1 =imag1[b:b+1,:,:,:]
        im2 =imag2[b:b+1,:,:,:]
        mse = torch.mean(torch.abs(im1 - im2) ** 2)
        if mse == 0:
            sum = sum + 100
        else:
            sum = sum + (10 * torch.log10(255 * 255 / mse))
    return sum / imag1.shape[0]

def cal_ssim(imag1,imag2):    #"b C H W"  tensor [0 ~ 1]
    # imag1 = imag1 * 255.0
    # imag2 = imag2 * 255.0
    ssim_sum = 0
    for i in range(imag1.shape[0]):
        im1 = imag1[i].permute(1,2,0)  #to 'H W C'
        im2 = imag2[i].permute(1,2,0)  #to 'H W C'
        im1 = im1.cpu().numpy()
        im2 = im2.cpu().numpy()
        # ssim = calculate_ssim(im1,im2)
        ssim = structural_similarity(im1, im2, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
        ssim_sum = ssim_sum + ssim
    return ssim_sum / imag1.shape[0]

# def ssim(img1, img2):
#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2
#     img1 = img1.astype(np.float32)
#     img2 = img2.astype(np.float32)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())

#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()

# def calculate_ssim(img1, img2):
#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1, img2))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError('Wrong input image dimensions.')


#-------------------------------------log ------------------------------------------
def get_timestamp():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=True):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root,"log/"+'MEF_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

#-------------------------------------log ------------------------------------------


if __name__ == '__main__':
    pass

