"""
This file contains the definition of the Dice-CE-Edge Loss for semantic segmentation tasks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Tumor SemSeg
from tumor_semseg.loss.utils import compute_dice, compute_iou, one_hot_encode


class SemSegLoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _forward(self, inputs: Tensor, targets: Tensor) -> dict[str, Tensor] | Tensor:
        pass

    def forward(self, inputs: Tensor, targets: Tensor) -> dict[str, Tensor]:
        loss = self._forward(inputs, targets)
        if isinstance(loss, dict):
            assert "total" in loss
            return loss
        elif isinstance(loss, Tensor):
            return {"total": loss}


class SimilatiryCoeffLoss(SemSegLoss):
    def __init__(self, compute_coeff: callable, weight: Optional[Tensor] = None, smooth: float = 1e-6):
        super().__init__()
        self.compute_coeff = compute_coeff
        self.weight = weight
        self.smooth = smooth

    def _forward(self, inputs: Tensor, targets: Tensor):
        coeff = self.compute_coeff(inputs, targets, self.smooth)
        return (1 - coeff).mean() if self.weight is None else (self.weight * (1 - coeff)).mean()


class DiceLoss(SimilatiryCoeffLoss):
    def __init__(self, weight: Optional[Tensor] = None, smooth: float = 1e-6):
        super().__init__(compute_dice, weight, smooth)


class IoULoss(SimilatiryCoeffLoss):
    def __init__(self, weight: Optional[Tensor] = None, smooth: float = 1e-6):
        super().__init__(compute_iou, weight, smooth)


class CELoss(SemSegLoss):
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def _forward(self, inputs: Tensor, targets: Tensor):
        if inputs.size(1) == 1:  # Number of classes
            return F.binary_cross_entropy_with_logits(inputs, targets, self.weight, reduction=self.reduction)
        else:
            return F.cross_entropy(inputs, targets.squeeze(1).long(), self.weight, self.reduction)


class EdgeLoss(SemSegLoss):
    def __init__(self, weight: Optional[Tensor] = None):
        super().__init__()
        self.weight = weight
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

    def _forward(self, inputs: Tensor, targets: Tensor):
        n_classes = inputs.size(1)
        targets_one_hot = one_hot_encode(targets, n_classes)
        loss = 0.0
        for class_idx in range(n_classes):
            class_input = inputs[:, class_idx : class_idx + 1, :, :]
            class_target = targets_one_hot[:, class_idx : class_idx + 1, :, :]
            edge_inputs = torch.abs(self.sobel_x(class_input)) + torch.abs(self.sobel_y(class_input))
            edge_targets = torch.abs(self.sobel_x(class_target)) + torch.abs(self.sobel_y(class_target))
            class_edge_loss = F.mse_loss(edge_inputs, edge_targets)
            loss += class_edge_loss if self.weight is None else self.weight[class_idx] * class_edge_loss

        return loss


class FocalLoss(SemSegLoss):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = CELoss(reduction="none")

        assert reduction in ["mean", "sum"]
        self.reduction = reduction

    def _forward(self, inputs: Tensor, targets: Tensor):
        ce_loss = self.ce_loss(inputs, targets)["total"]
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)


class TverskyLoss(SemSegLoss):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def _forward(self, inputs, targets):
        n_batches = inputs.size(0)
        n_classes = inputs.size(1)
        inputs = inputs.sigmoid() if n_classes == 1 else inputs.softmax(dim=1)
        inputs = inputs.view(n_batches, n_classes, -1)

        targets_one_hot = one_hot_encode(targets, n_classes)
        targets_one_hot = targets_one_hot.view(n_batches, n_classes, -1)

        tp = (inputs * targets_one_hot).sum(dim=-1)
        fp = ((1 - targets_one_hot) * inputs).sum(dim=-1)
        fn = (targets_one_hot * (1 - inputs)).sum(dim=-1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return 1 - tversky.mean()


class FocalTverskyLoss(FocalLoss):
    def __init__(
        self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean", tversky: TverskyLoss = TverskyLoss()
    ):
        super().__init__(alpha, gamma, reduction)
        self.ce_loss = tversky


@dataclass
class DiceFocalEdgeLossConfig:
    dice: DiceLoss = DiceLoss()
    focal: FocalLoss = FocalLoss()
    edge: EdgeLoss = EdgeLoss()
    dice_weight: Optional[Tensor] = None
    focal_weight: Optional[Tensor] = None
    edge_weight: Optional[Tensor] = None


class DiceFocalEdgeLoss(SemSegLoss):
    def __init__(self, config: DiceFocalEdgeLossConfig = DiceFocalEdgeLossConfig()):
        super().__init__()
        self.dice_weight = config.dice_weight
        self.focal_weight = config.focal_weight
        self.edge_weight = config.edge_weight
        self.dice_loss = config.dice
        self.focal_loss = config.focal
        self.edge_loss = config.edge

    def _forward(self, inputs: Tensor, targets: Tensor):
        dice_loss = self.dice_loss(inputs, targets)["total"]
        focal_loss = self.focal_loss(inputs, targets)["total"]
        edge_loss = self.edge_loss(inputs, targets)["total"]
        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss + self.edge_weight * edge_loss

        return {"total": total_loss, "dice": dice_loss, "focal": focal_loss, "edge": edge_loss}
