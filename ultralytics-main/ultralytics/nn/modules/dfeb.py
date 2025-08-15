import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
IN=64

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DWConv, self).__init__()
        # 深度卷积：每个输入通道对应一个卷积核
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,  # 关键参数：分组数等于输入通道数
            bias=False
        )
        # 逐点卷积：1x1卷积融合通道信息
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 使用示例


class LRB(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.DWconv1 = DWConv(in_channels, in_channels, kernel_size=5, stride=1, padding=2)
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.DWconv2 = DWConv(in_channels, in_channels, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        copyx=x.clone()
        x = self.conv1(x)
        x = self.DWconv1(x)
        if x.shape[-1]!=1:
            x = self.batchnorm1(x)
        x = self.silu(x)
        x = self.conv2(x)
        x = self.DWconv2(x)
        if x.shape[-1]!=1:
            x = self.batchnorm2(x)
        x = self.silu(x)
        x+=copyx
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, shift_distance=1):
        super(DConv, self).__init__()
        self.shift_distance = shift_distance
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # 标准卷积权重
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        B, C, H, W = x.shape
        out_channels = self.weight.shape[0]
        device = x.device
        shift = self.shift_distance

        # 定义四个方向偏移
        shifts = [(0, 0), (0, shift), (shift, 0), (0, -shift), (-shift, 0)]
        idx = (torch.arange(C) % 5).to(device)

        # 创建偏移后的输入
        shifted_inputs = []
        x=F.pad(x, (self.shift_distance,)*4,mode='constant', value=0)
        for c in range(C):
            dx, dy = shifts[idx[c]]
            x_c = x[:, c:c+1, :, :]
            if dx != 0 or dy != 0:
                x_c = torch.roll(x_c, shifts=(dy, dx), dims=(2, 3))
            shifted_inputs.append(x_c)
        x_shifted = torch.cat(shifted_inputs, dim=1)

        # 执行标准卷积
        out = F.conv2d(x_shifted, self.weight, self.bias, self.stride, self.padding, self.dilation)
        return out[:,:,1:-1,1:-1]

class DDWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, shift_distance=1):
        super(DDWConv, self).__init__()
        self.shift_distance = shift_distance
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # 标准卷积权重
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        B, C, H, W = x.shape
        out_channels = self.weight.shape[0]
        device = x.device
        shift = self.shift_distance

        # 定义四个方向偏移
        shifts = [(0, 0), (0, shift), (shift, 0), (0, -shift), (-shift, 0)]
        idx = (torch.arange(C) % 5).to(device)

        # 创建偏移后的输入
        shifted_inputs = []
        x=F.pad(x, (self.shift_distance,)*4,mode='constant', value=0)
        for c in range(C):
            dx, dy = shifts[idx[c]]
            x_c = x[:, c:c+1, :, :]
            if dx != 0 or dy != 0:
                x_c = torch.roll(x_c, shifts=(dy, dx), dims=(2, 3))
            shifted_inputs.append(x_c)
        x_shifted = torch.cat(shifted_inputs, dim=1)

        # 执行标准卷积
        out = F.conv2d(x_shifted, self.weight, self.bias, self.stride, self.padding, self.dilation,groups=C)
        return out[:,:,1:-1,1:-1]

class LEB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        if in_channels % 2 != 0:
            print("LEB输入不是2的倍数")
        self.DDWconv = DDWConv(in_channels, in_channels)
        self.DWconv1 = DWConv(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1)
        self.DWconv2 = DWConv(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1)
        self.DConv = DConv(in_channels*2,1)
        self.GELU = nn.GELU()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=3, padding=1)
    def forward(self, x):
        in_channels = self.in_channels
        xcopy=x.clone()
        xf,xs=torch.split(x,in_channels//2,dim=1)
        xf=self.upsample1(self.DWconv1(self.conv1(xf)))
        xs=self.upsample2(self.DWconv2(self.conv2(xs)))
        xf=xf[:,:,0:x.shape[2],0:x.shape[3]]
        xs=xs[:,:,0:x.shape[2],0:x.shape[3]]
        x_re = torch.concatenate((xf,xs,self.DDWconv(xcopy)), dim=1)
        return self.GELU(self.DConv(x_re))

class DFEB(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels*2+1, in_channels, kernel_size=1, stride=1, padding=0)
        self.LRB = LRB(in_channels)
        self.LEB = LEB(in_channels)
    def forward(self, x):
        in_channels = self.in_channels
        xf,xs=torch.split(self.conv1(x),in_channels,dim=1)
        xfcopy=xf.clone()
        x_re=torch.cat((self.LRB(xf),xs,self.LEB(xfcopy)), dim=1)
        return self.conv2(x_re)
