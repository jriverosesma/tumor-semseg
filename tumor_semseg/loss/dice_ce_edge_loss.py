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
    def __init__(self, n_classes, weight=None, smooth=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.weight = weight

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        inputs = inputs.view(inputs.size(0), self.n_classes, -1)

        targets_one_hot = F.one_hot(targets.long(), self.n_classes).squeeze().permute(0, 3, 1, 2).float()
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), self.n_classes, -1)

        intersection = (inputs * targets_one_hot).sum(-1)
        union = inputs.sum(-1) + targets_one_hot.sum(-1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return (1 - dice).mean() if self.weight is None else (self.weight * (1 - dice)).mean()


class CrossEntropyLoss(nn.Module):
    def __init__(self, n_classes, weight=None):
        super().__init__()
        self.n_classes = n_classes
        self.weight = weight

    def forward(self, inputs, targets):
        one_hot_targets = F.one_hot(targets.long(), self.n_classes).transpose(1, 4).squeeze().float()

        return F.cross_entropy(inputs, one_hot_targets, self.weight)


class EdgeLoss(nn.Module):
    def __init__(self, n_classes, weight=None):
        super().__init__()
        self.weight = weight
        self.n_classes = n_classes
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

    def forward(self, inputs, targets):
        targets_one_hot = F.one_hot(targets.long(), self.n_classes).squeeze().permute(0, 3, 1, 2).float()
        loss = 0.0
        for class_idx in range(self.n_classes):
            class_input = inputs[:, class_idx : class_idx + 1, :, :]
            class_target = targets_one_hot[:, class_idx : class_idx + 1, :, :]
            edge_inputs = torch.abs(self.sobel_x(class_input)) + torch.abs(self.sobel_y(class_input))
            edge_targets = torch.abs(self.sobel_x(class_target)) + torch.abs(self.sobel_y(class_target))
            class_edge_loss = F.mse_loss(edge_inputs, edge_targets)
            loss += class_edge_loss if self.weight is None else self.weight[class_idx] * class_edge_loss

        return loss


@dataclass
class DiceCEEdgeLossConfig:
    n_classes: int
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.1
    dice_smooth: float = 1e-6
    dice_weight: Optional[Tensor] = None
    ce_weight: Optional[Tensor] = None
    edge_weight: Optional[Tensor] = None


class DiceCEEdgeLoss(nn.Module):
    def __init__(self, config: DiceCEEdgeLossConfig):
        super().__init__()
        self.config = config
        self.dice_loss = DiceLoss(config.n_classes, config.dice_weight, config.dice_smooth)
        self.cross_entropy_loss = CrossEntropyLoss(config.n_classes, config.ce_weight)
        self.edge_loss = EdgeLoss(config.n_classes, config.edge_weight)

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        ce_loss = self.cross_entropy_loss(inputs, targets)
        edge_loss = self.edge_loss(inputs, targets)
        total_loss = self.config.alpha * dice_loss + self.config.beta * ce_loss + self.config.gamma * edge_loss

        return {"total": total_loss, "dice": dice_loss, "ce": ce_loss, "edge": edge_loss}
