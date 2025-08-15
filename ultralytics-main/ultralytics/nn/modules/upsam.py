import torch
import torch.nn as nn
import torch.nn.functional as F


class UM(nn.Module):
    """
    上采样多尺度学习模块 - 通过转置卷积把特征图放大
    """

    def __init__(self, in_channels, out_channels, scale):
        """
        参数
        ----
        in_channels  : 输入特征图的通道数
        out_channels : 输出特征图的通道数
        scale        : 上采样倍数（必须 >= 1 的整数）
        """
        super(UM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale

        # 转置卷积（反卷积）实现上采样
        self.up_conv = nn.ConvTranspose2d(
            in_channels=in_channels,  # 输入通道
            out_channels=out_channels,  # 输出通道
            kernel_size=4,
            stride=2,  # 2 倍上采样
            padding=1,
            output_padding=0,
            bias=False
        )
    def forward(self, x):
        x = self.up_conv(x)
        return x