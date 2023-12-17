import pytest
import torch

# Tumor SemSeg
from tumor_semseg.loss.semseg_losses import (
    CELoss,
    DiceFocalEdgeLoss,
    DiceLoss,
    EdgeLoss,
    FocalLoss,
    FocalTverskyLoss,
    IoULoss,
    TverskyLoss,
)


@pytest.mark.parametrize(
    "loss_fn",
    [
        (CELoss),
        (DiceFocalEdgeLoss),
        (DiceLoss),
        (EdgeLoss),
        (FocalLoss),
        (FocalTverskyLoss),
        (IoULoss),
        (TverskyLoss),
    ],
)
def test_loss(loss_fn):
    compute_loss = loss_fn()

    # Binary
    inputs = torch.ones(5, 1, 256, 256)
    targets = torch.ones(5, 1, 256, 256)
    assert pytest.approx(compute_loss(inputs, targets)["total"]) == 0

    # Multiclass
    # inputs = torch.zeros(5, 3, 256, 256)
    # targets = torch.zeros(5, 1, 256, 256)
    # inputs[:, 0, ...] = 1
    # assert pytest.approx(compute_loss(inputs, targets)["total"]) == 0
