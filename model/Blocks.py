import math
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import numpy as np
import torch.nn.init as init

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

#=============================================ResBlock=========================================
class ResBlock_IN(nn.Module):
    def __init__(self,in_size,out_size):
        super(ResBlock_IN,self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.norm = nn.InstanceNorm2d(out_size//2, affine=True)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride = 1, padding=1, bias=True)
        self.relu_1 = nn.Sequential( nn.LeakyReLU(0.2, inplace=False), )
        self.conv_2 = nn.Sequential( nn.Conv2d(out_size, out_size, kernel_size=3, stride = 1, padding=1, bias=True), nn.LeakyReLU(0.2, inplace=False),)

    def forward(self, x):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.conv_2(out)
        out += self.identity(x)
        return out 
