"""
This file contains the definition of the Dice-CE-Edge Loss for semantic segmentation tasks.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiceLoss(nn.Module):
    def __init__(self, smooth):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, self.weight)


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y)

    def forward(self, inputs, targets):
        edge_inputs = torch.abs(self.sobel_x(inputs)) + torch.abs(self.sobel_y(inputs))
        edge_targets = torch.abs(self.sobel_x(targets)) + torch.abs(self.sobel_y(targets))
        return F.mse_loss(edge_inputs, edge_targets)


@dataclass
class DiceCEEdgeLossConfig:
    n_classes: int
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.1
    dice_smooth: float = 1e-6
    ce_weight: Optional[Tensor] = None


class DiceCEEdgeLoss(nn.Module):
    def __init__(self, config: DiceCEEdgeLossConfig):
        super().__init__()
        self.config = config
        self.dice_loss = DiceLoss(config.dice_smooth)
        self.cross_entropy_loss = CrossEntropyLoss(config.ce_weight)
        self.edge_loss = EdgeLoss()

    def forward(self, inputs, targets):
        # one_hot_inputs = F.one_hot(inputs.view(-1), self.config.n_classes)
        dice_loss = self.dice_loss(inputs, targets)
        cross_entropy_loss = self.cross_entropy_loss(inputs, targets)
        edge_loss = self.edge_loss(inputs, targets)
        total_loss = (
            self.config.alpha * dice_loss + self.config.beta * cross_entropy_loss + self.config.gamma * edge_loss
        )
        return total_loss
