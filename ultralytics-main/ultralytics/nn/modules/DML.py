import torch
import torch.nn as nn
import torch.nn.functional as F


class DML(nn.Module):
    """
    深度多尺度学习模块 - 融合不同尺度的特征信息
    """

    def __init__(self, in_channels, out_channels):
        """
        初始化DML模块

        参数:
            in_channels: 输入特征图的通道数
            out_channels: 输出特征图的通道数
            pool_sizes: 池化核大小列表，默认为[5, 7]
            norm_layer: 归一化层类型，默认为BatchNorm2d
        """
        super(DML, self).__init__()

        # 保存输入参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    def forward(self, x):
        x = self.maxpool1(x)
        x = self.maxpool2(x)
        x = self.conv1(x)
        return x