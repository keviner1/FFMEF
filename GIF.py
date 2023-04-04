from math import exp
import torch.nn.functional as F
import torch.nn as nn
import torch

from einops import rearrange

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

class GIF_Loss(nn.Module):
    def __init__(self):
        super(GIF_Loss, self).__init__()
        self.l1 = L1Loss_W()
        self.l2 = L2Loss_W()

    def window(self, x, windows, stride, padding, opt):
        b, c, h, w = x.shape
        kx=F.unfold(x, kernel_size=windows, stride=stride, padding=padding) #b, k*k, n_H*n_W
        kx = rearrange(kx, "b c (h w) -> b c h w", h = h, w = w)
        if opt == "avg":
            kx = torch.mean(kx, 1, keepdim=True)
        if opt == "max":
            kx,i = torch.max(kx, 1, keepdim=True)
        if opt == "min":
            kx,i = torch.min(kx, 1, keepdim=True)
        return kx

    def Gmask(self, x, threshold):
        mask = x >= threshold
        x[mask] = 1
        mask = x < threshold
        x[mask] = 0
        return x

    def scharr(self, x):
        b, c, h ,w = x.shape
        pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
        x = pad(x)
        kx=F.unfold(x,kernel_size=3,stride=1,padding=0) #b,n*k*k,n_H*n_W
        kx=kx.permute([0,2,1]) #b,n_H*n_W,n*k*k
        # kx=kx.view(1, b*h*w, 9) #1,b*n_H*n_W,n*k*k

        w1 = torch.tensor([-3, 0, 3, -10, 0, 10, -3, 0, 3]).float().cuda()
        w2 = torch.tensor([-3, -10, -3, 0, 0 ,0, 3, 10, 3]).float().cuda()

        y1=torch.matmul(kx*255.0,w1) #1,b*n_H*n_W,1
        y2=torch.matmul(kx*255.0,w2) #1,b*n_H*n_W,1
        # y1=y1.view(b,h*w,1) #b,n_H*n_W,1
        y1=y1.unsqueeze(-1).permute([0,2,1]) #b,1,n_H*n_W
        # y2=y2.view(b,h*w,1) #b,n_H*n_W,1
        y2=y2.unsqueeze(-1).permute([0,2,1]) #b,1,n_H*n_W

        y1=F.fold(y1,output_size=(h,w),kernel_size=1) #b,m,n_H,n_W
        y2=F.fold(y2,output_size=(h,w),kernel_size=1) #b,m,n_H,n_W
        y1 = y1.clamp(-255, 255)
        y2 = y2.clamp(-255, 255)
        return (0.5*torch.abs(y1) + 0.5*torch.abs(y2))/255.0

    def softmax(self, x1, x2, a):
        x1 = torch.exp(x1*a)/(torch.exp(x1*a) + torch.exp(x2*a))
        x2 = torch.exp(x2*a)/(torch.exp(x1*a) + torch.exp(x2*a))
        return x1, x2

    def forward(self, y0, g0, y1, g1, y0_Cb, y0_Cr, y1_Cb, y1_Cr, fused, vis):
        g_max,i = torch.max(torch.cat([g0,g1],1),1)
        g_max = g_max.unsqueeze(1)
        g_fus = self.scharr(fused)

        y_avg = (y0+y1) / 2
        weight_0,weight_1 = self.softmax( torch.abs(y0_Cb - 0.5) + torch.abs(y0_Cr - 0.5), torch.abs(y1_Cb - 0.5) + torch.abs(y1_Cr - 0.5),1)  
        weight_0,weight_1 = self.softmax(weight_0, weight_1,1)
        weighted_avg = weight_0*y0 + weight_1*y1

        g0_max3 = self.window(g0, 3, 1, 1, "max")
        g1_max3 = self.window(g1, 3, 1, 1, "max")
        g0_max3_01m = self.Gmask(g0_max3, 0.5)
        g1_max3_01m = self.Gmask(g1_max3, 0.5)
        g_smooth = 1- (g0_max3_01m + g1_max3_01m).clamp(0,1)

        loss_gra_guided = self.l1(y0, fused, g0_max3_01m) + self.l1(y1, fused, g1_max3_01m)
        loss_gra_else = self.l1(weighted_avg, fused, g_smooth)
        loss_gra_max = self.l1(g_max, g_fus, 1)
        loss = loss_gra_guided*1.25 + loss_gra_else + loss_gra_max*2

        return loss

