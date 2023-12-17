import pytest
import torch

# Tumor SemSeg
from tumor_semseg.loss.semseg_losses import (
    CELoss,
    DiceFocalEdgeLoss,
    DiceLoss,
    FocalLoss,
    FocalTverskyLoss,
    IoULoss,
    TverskyLoss,
)


@pytest.mark.parametrize(
    "loss_fn",
    [
        (DiceLoss),
        (IoULoss),
        (CELoss),
        (FocalLoss),
        (FocalTverskyLoss),
        (TverskyLoss),
        (DiceFocalEdgeLoss),
    ],
)
def test_loss(loss_fn):
    inputs = torch.ones(5, 3, 256, 256)
    targets = torch.ones(5, 1, 256, 256)
    compute_loss = loss_fn()

    assert pytest.approx(compute_loss(inputs, targets)["total"]) == 0
