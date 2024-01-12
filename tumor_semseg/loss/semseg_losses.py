"""
This file contains the definition of the Dice-CE-Edge Loss for semantic segmentation tasks.
"""

# Standard
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

# Third-Party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# TumorSemSeg
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
        input_probs = inputs.sigmoid() if inputs.size(1) == 1 else inputs.softmax(dim=1)
        coeff = self.compute_coeff(input_probs, targets, **self.params)
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
                inputs, targets.unsqueeze(1), pos_weight=self.weight, reduction=self.reduction
            )
        else:
            return F.cross_entropy(inputs, targets.long(), weight=self.weight, reduction=self.reduction)


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

        if n_classes == 1:
            log_pt = F.logsigmoid(inputs)
            pt = torch.sigmoid(inputs)
        else:
            log_pt = F.log_softmax(inputs, dim=1).view(n_batches, n_classes, -1)
            pt = F.softmax(inputs, dim=1).view(n_batches, n_classes, -1)

        focal_term = (1 - pt) ** self.gamma

        if n_classes == 1:
            focal_loss = -self.alpha * focal_term * (targets * log_pt + (1 - targets) * torch.log(1 - pt))
        else:
            targets_one_hot = one_hot_encode(targets, n_classes).view(n_batches, n_classes, -1)
            focal_loss = -self.alpha * focal_term * (targets_one_hot * log_pt)

        focal_loss = reduce(focal_loss, dims=[-1], reduction=self.class_reduction)
        focal_loss = reduce(focal_loss, reduction=self.batch_reduction)

        return focal_loss


class ComposedLoss(SemSegLoss):
    def __init__(self, losses: list[SemSegLoss], weight: Optional[Tensor] = None):
        super().__init__()
        self.losses = losses
        self.weight = weight

    def _forward(self, inputs: Tensor, targets: Tensor):
        loss = {}
        loss["total"] = 0.0
        for i, compute_loss in enumerate(self.losses):
            loss_name = compute_loss.__class__.__name__.lower()
            loss[loss_name] = compute_loss(inputs, targets)["total"]
            loss["total"] += loss[loss_name] if self.weight is None else self.weight[i] * loss[loss_name]

        return loss
