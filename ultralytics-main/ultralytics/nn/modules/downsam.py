import torch
import torch.nn as nn
import torch.nn.functional as F


class DM(nn.Module):
    """
    深度多尺度学习模块 - 融合不同尺度的特征信息
    """

    def __init__(self, in_channels, out_channels,scale):
        super(DM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=scale,padding=scale//2)
    def forward(self, x):
        x=self.conv(x)
        return x
