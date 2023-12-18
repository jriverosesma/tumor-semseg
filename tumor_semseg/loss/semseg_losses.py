"""
This file contains the definition of the Dice-CE-Edge Loss for semantic segmentation tasks.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Tumor SemSeg
from tumor_semseg.loss.utils import compute_dice, compute_iou, compute_tversky, one_hot_encode, reduce


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
    def __init__(
        self,
        compute_coeff: Callable,
        params: dict[str, Any] = {},
        weight: Optional[Tensor] = None,
        class_reduction: str = "mean",
        batch_reduction: str = "mean",
    ):
        super().__init__()
        self.compute_coeff = compute_coeff
        self.params = params
        self.weight = weight
        self.class_reduction = class_reduction
        self.batch_reduction = batch_reduction

    def _forward(self, inputs: Tensor, targets: Tensor):
        coeff = self.compute_coeff(inputs, targets, **self.params)
        loss = (1 - coeff) if self.weight is None else self.weight * (1 - coeff)

        loss = reduce(loss, dims=[-1], reduction=self.class_reduction)
        loss = reduce(loss, reduction=self.batch_reduction)

        return loss


class DiceLoss(SimilatiryCoeffLoss):
    def __init__(
        self,
        params: dict[str, Any] = {},
        weight: Optional[Tensor] = None,
        class_reduction: str = "none",
        batch_reduction: str = "mean",
    ):
        super().__init__(compute_dice, params, weight, class_reduction, batch_reduction)


class IoULoss(SimilatiryCoeffLoss):
    def __init__(
        self,
        params: dict[str, Any] = {},
        weight: Optional[Tensor] = None,
        class_reduction: str = "none",
        batch_reduction: str = "mean",
    ):
        super().__init__(compute_iou, params, weight, class_reduction, batch_reduction)


class TverskyLoss(SimilatiryCoeffLoss):
    def __init__(
        self,
        params: dict[str, Any] = {},
        weight: Optional[Tensor] = None,
        class_reduction: str = "none",
        batch_reduction: str = "mean",
    ):
        super().__init__(compute_tversky, params, weight, class_reduction, batch_reduction)


class CELoss(SemSegLoss):
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def _forward(self, inputs: Tensor, targets: Tensor):
        if inputs.size(1) == 1:  # Number of classes
            return F.binary_cross_entropy_with_logits(
                inputs, targets.unsqueeze(1), self.weight, reduction=self.reduction
            )
        else:
            return F.cross_entropy(inputs, targets.long(), self.weight, reduction=self.reduction)


class FocalLoss(SemSegLoss):
    def __init__(
        self, alpha: float = 1.0, gamma: float = 2.0, class_reduction: str = "none", batch_reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_reduction = class_reduction
        self.batch_reduction = batch_reduction

    def _forward(self, inputs: Tensor, targets: Tensor):
        n_batches = inputs.size(0)
        n_classes = inputs.size(1)

        log_pt = F.logsigmoid(inputs) if n_classes == 1 else F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)
        focal_term = (1 - pt) ** self.gamma
        focal_loss = -self.alpha * focal_term * log_pt

        focal_loss = focal_loss.view(n_batches, n_classes, -1)
        targets_one_hot = one_hot_encode(targets, n_classes).view(n_batches, n_classes, -1)

        focal_loss = focal_loss * targets_one_hot

        focal_loss = reduce(focal_loss, dims=[-1], reduction=self.class_reduction)
        focal_loss = reduce(focal_loss, reduction=self.batch_reduction)

        return focal_loss
