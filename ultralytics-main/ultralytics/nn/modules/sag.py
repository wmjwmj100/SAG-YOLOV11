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




import torch
import torch.nn as nn
import torch.nn.functional as F



from typing import List

class SAG(nn.Module):
    def __init__(self, in1_channels: int, in2_channels: int):
        super().__init__()
        # 所有通道数在 init 时确定
        self.conv1 = nn.Conv2d(in1_channels, in2_channels, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(in2_channels, in2_channels, 1, 1, 0, bias=False)
        self.dwconv  = DWConv(in2_channels, in2_channels, 3, 1, 1)
        self.bn      = nn.BatchNorm2d(in2_channels)
        self.conv3   = nn.Conv2d(in2_channels, 1, 3, 2, 1, bias=False)   # 下采样 mask
        self.act     = nn.SiLU()

    def forward(self, x: List[torch.Tensor]):
        x1=x[0]
        x22=x[1]
        # x1: [B, C1, H,  W  ]
        # x2: [B, C2, H/2, W/2]
        smask = self.act(self.conv3(self.conv1(x1)))        # [B,1,H/2,W/2]
        x2_re = self.bn(self.dwconv(self.conv2(x22)))        # [B,C2,H/2,W/2]
        x2_out = x2_re * smask                            # 逐通道加权
        return x2_out + x22