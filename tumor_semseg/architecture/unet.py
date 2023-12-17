"""
UNet model architecture definition.
Mainly based on UNet implementation of: https://github.com/milesial/Pytorch-UNet
"""

from dataclasses import dataclass

import torch.nn as nn

# Tumor SemSeg
from tumor_semseg.architecture.unet_blocks import DoubleConv, Down, Head, Up


@dataclass
class UNetConfig:
    n_classes: int
    in_channels: int = 3
    activation: nn.Module = nn.ReLU(inplace=True)
    bilinear: bool = False


class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.in_conv = DoubleConv(config.in_channels, 32, config.activation)
        self.down_conv_1 = Down(32, 64, config.activation)
        self.down_conv_2 = Down(64, 128, config.activation)
        self.down_conv_3 = Down(128, 256, config.activation)
        self.down_conv_4 = Down(256, 512, config.activation)
        self.up_conv1 = Up(512, 256, config.activation, config.bilinear)
        self.up_conv2 = Up(256, 128, config.activation, config.bilinear)
        self.up_conv3 = Up(128, 64, config.activation, config.bilinear)
        self.up_conv4 = Up(64, 32, config.activation, config.bilinear)
        self.head = Head(32, config.n_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down_conv_1(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.down_conv_3(x3)
        x = self.down_conv_4(x4)
        x = self.up_conv1(x, x4)
        x = self.up_conv2(x, x3)
        x = self.up_conv3(x, x2)
        x = self.up_conv4(x, x1)

        return self.head(x)
