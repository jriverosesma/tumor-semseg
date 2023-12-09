"""
UNet model architecture blocks definition.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: nn.Module):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: nn.Module):
        super().__init__()
        self.down_conv = nn.Sequential(nn.MaxPool2d(kernel_size=2), DoubleConv(in_channels, out_channels, activation))

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: nn.Module):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, activation)

    def forward(self, x1, x2):
        x = self.up(x1)
        # NOTE: same convolutions ensure that the down/upwards path features have the same dimensions
        # No need for extra padding here
        x = torch.cat([x2, x], dim=1)
        return self.conv(x)


class Head(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        return self.head(x)
