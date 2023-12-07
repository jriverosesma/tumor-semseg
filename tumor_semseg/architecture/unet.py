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


class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()

        self.in_conv = DoubleConv(config.in_channels, 64, config.activation)
        self.down_conv_1 = Down(64, 128, config.activation)
        self.down_conv_2 = Down(128, 256, config.activation)
        self.down_conv_3 = Down(256, 512, config.activation)
        self.down_conv_4 = Down(512, 1024, config.activation)
        self.up_conv1 = Up(1024 + 512, 512, config.activation)
        self.up_conv2 = Up(512 + 256, 256, config.activation)
        self.up_conv3 = Up(256 + 128, 128, config.activation)
        self.up_conv4 = Up(128 + 64, 64, config.activation)
        self.head = Head(64, config.n_classes)

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
