from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .Blocks import ResBlock_IN
import numpy as np
import cv2
import pywt
import torch.nn.init as init

class FFMEF(nn.Module):
    def __init__(self, channels):
        super(FFMEF, self).__init__()
        self.channels = channels
        self.preconv =  nn.Conv2d(1, channels, 1, 1, 0)
        self.encode = HUnet(channels)

        self.kp1 = KPN(channels, 1, 1, 1, stride=1, padding=0)
        self.kp2 = KPN(channels, 1, 1, 3, stride=1, padding=1)
        self.kp3 = KPN(channels, 1, 1, 5, stride=1, padding=2)

        self.sca1 = SpatialCrossAttention(channels)
        self.sca2 = SpatialCrossAttention(channels)
        self.sca3 = SpatialCrossAttention(channels)

        self.alpha1 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.alpha1.data.fill_(0.6)
        self.alpha2 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.alpha2.data.fill_(0.3)
        self.alpha3 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.alpha3.data.fill_(0.1)
        self.apply(initialize_weights_xavier)

    def SFFusion(self, I0, I1, Flist_0, Flist_1):
        #level1
        Flist_0_2, Flist_1_2 = self.sca1(Flist_0[2], Flist_1[2])
        I0_1 = self.kp1(I0, Flist_0_2)
        I1_1 = self.kp1(I1, Flist_1_2)
        If_1 = I0_1 + I1_1
        #level2
        Flist_0_1, Flist_1_1 = self.sca2(Flist_0[1], Flist_1[1])
        I0_2 = self.kp2(I0, Flist_0_1)
        I1_2 = self.kp2(I1, Flist_1_1)
        If_2 = I0_2 + I1_2
        # #level3
        Flist_0_0, Flist_1_0 = self.sca3(Flist_0[0], Flist_1[0])
        I0_3 = self.kp3(I0, Flist_0_0)
        I1_3 = self.kp3(I1, Flist_1_0)
        If_3 = I0_3 + I1_3
        If = self.alpha1 * If_1 + self.alpha2 * If_2 + self.alpha3 * If_3
        return If

    def forward(self, img0_RGB, img0_Y, img1_RGB, img1_Y):

        f_0 = self.preconv(img0_Y)
        f_1 = self.preconv(img1_Y)

        #HRE
        Flist_0 = self.encode(f_0)
        Flist_1 = self.encode(f_1)

        #SFP + F-Fusion
        I_fus = self.SFFusion(img0_Y, img1_Y, Flist_0, Flist_1)

        return I_fus

class SpatialCrossAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialCrossAttention, self).__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(channels, 2, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1,x2],1)
        attn = self.spatial_attention(x)
        attn1, attn2 = torch.chunk(attn, 2, 1)
        return x1*attn1, x2*attn2
    

class KPN(nn.Module):
    def __init__(self, condition_channels, channels_in, channels_out, kernel_size, stride=1, padding=1, dilation=1, groups=1, use_bias=True):
        super(KPN, self).__init__()
        self.condition_channels = condition_channels
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias

        self.spatial_attention=nn.Sequential(
            nn.Conv2d(condition_channels, condition_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(condition_channels, kernel_size**2, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size**2, kernel_size**2, 1),
        )

    def forward(self, x, C):
        (b, n, H, W) = x.shape
        m=self.channels_out
        k=self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)

        #--------------spatial---------------------
        atw1=self.spatial_attention(C) #b,k*k,n_H,n_W
        atw1=atw1.permute([0,2,3,1]) #b,n_H,n_W,k*k
        atw1 = atw1.repeat([1,1,1,n])
        # atw1 = atw1 * atw2  #spectral

        atw=atw1
        atw=atw.view(b,n_H*n_W,n*k*k) #b,n_H*n_W,n*k*k
        atw=atw.permute([0,2,1]) #b,n*k*k,n_H*n_W

        kx=F.unfold(x,kernel_size=k,stride=self.stride,padding=self.padding) #b,n*k*k,n_H*n_W
        atx=atw*kx #b,n*k*k,n_H*n_W
        y = torch.sum(atx, 1)
        y = y.unsqueeze(1)
        y=F.fold(y,output_size=(n_H,n_W),kernel_size=1) #b,m,n_H,n_W
        return y

class HUnet(nn.Module):
    def __init__(self, channels):
        super(HUnet, self).__init__()
        #---------ENCODE
        self.layer_dowm1 = basic_block(channels,channels,"RES")
        self.dowm1 = sample_block(channels, channels, "DOWN", 1)
        self.layer_dowm2 = basic_block(channels,channels,"RES")
        self.dowm2 = sample_block(channels, channels, "DOWN", 1)
        #---------DECODE
        self.layer_bottom = basic_block(channels,channels,"RES")
        self.up2 = sample_block(channels, channels, "UP", 1)
        self.layer_up2 = basic_block(channels,channels,"RES")
        self.up1 = sample_block(channels, channels, "UP", 1)
        self.layer_up1 = basic_block(channels,channels,"RES")
        #---------SKIP
        self.fus2 = skip(channels*2, channels, "RES")
        self.fus1 = skip(channels*2, channels, "RES")
        #---------SKIP
        self.skip1 = skip(channels*2, channels, "CONV")
        self.skip2 = skip(channels*2, channels, "CONV")
        self.skip3 = skip(channels*2, channels, "CONV")
        self.skip4 = skip(channels*2, channels, "CONV")
        self.skip5 = skip(channels*2, channels, "CONV")
        self.skip6 = skip(channels*2, channels, "CONV")

    def forward(self, x):
        #---------ENCODE
        x_11 = self.layer_dowm1(x)
        x_down1 = self.dowm1(x_11)
        x_down1 = self.skip1(torch.cat([x,x_down1],1), x_down1)

        x_12 = self.layer_dowm2(x_down1)
        x_down2 = self.dowm2(x_12)
        x_down2 = self.skip2(torch.cat([x_down1,x_down2],1), x_down2)
        x_down2 = self.skip3(torch.cat([x,x_down2],1), x_down2)

        x_bottom = self.layer_bottom(x_down2)

        #---------DECODE
        x_up2 = self.up2(x_bottom)
        x_up2 = self.skip4(torch.cat([x_bottom,x_up2],1), x_up2)
        x_22 = self.layer_up2(x_up2)
        x_22 = self.fus2(torch.cat([x_12,x_22],1), x_22)

        x_up1 = self.up1(x_22)
        x_up1 = self.skip5(torch.cat([x_22,x_up1],1), x_up1)
        x_21 = self.layer_up1(x_up1)
        x_21 = self.skip6(torch.cat([x_bottom,x_21],1), x_21)
        x_21 = self.fus1(torch.cat([x_11,x_21],1), x_21)
        return [x_bottom, x_22, x_21]

class skip(nn.Module):
    def __init__(self, channels_in, channels_out, block):
        super(skip, self).__init__()
        if block == "CONV":
            self.body = nn.Sequential(nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=True),
                                        nn.InstanceNorm2d(channels_out, affine=True),nn.ReLU(inplace = True),)
        if block == "ID":
            self.body = nn.Identity()
        if block == "RES":
            self.body = nn.Sequential(ResBlock_IN(channels_in, channels_out)) 
        #--------------------------------------
        self.alpha1 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.alpha1.data.fill_(1.0)
        self.alpha2 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.alpha2.data.fill_(0.5)

    def forward(self, x, y):
        out = self.alpha1 * self.body(x) + self.alpha2 * y
        return out

class sample_block(nn.Module):
    def __init__(self, channels_in, channels_out, size, dil):
        super(sample_block, self).__init__()
        #------------------------------------------
        if size == "DOWN":
            self.conv = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, 1, dil, dilation = dil),
                nn.InstanceNorm2d(channels_out, affine=True),
                nn.ReLU(inplace=True),
            )
        if size == "UP":
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(channels_in, channels_out, 3, 1, dil, dilation = dil),
                nn.InstanceNorm2d(channels_out, affine=True),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        return self.conv(x)

class basic_block(nn.Module):
    def __init__(self, channels_in, channels_out, block):
        super(basic_block, self).__init__()
        #------------------------------------------
        if block == "CONV":
            self.body = nn.Sequential(nn.Conv2d(channels_in, channels_out, 3, 1, 1, bias=True),
                                        nn.InstanceNorm2d(channels_out, affine=True),nn.ReLU(inplace = True),)
        if block == "RES":
            self.body = nn.Sequential(ResBlock_IN(channels_in, channels_out))

    def forward(self, x):
        return self.body(x)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)