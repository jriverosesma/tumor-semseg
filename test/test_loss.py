# Third-Party
import pytest
import torch

# TumorSemSeg
from tumor_semseg.loss.semseg_losses import CELoss, ComposedLoss, DiceLoss, FocalLoss, IoULoss, TverskyLoss


@pytest.mark.parametrize(
    "loss_fn",
    [(CELoss), (DiceLoss), (FocalLoss), (IoULoss), (TverskyLoss)],
)
def test_single_loss(loss_fn):
    compute_loss = loss_fn()

    # Binary
    inputs = 10 * torch.ones(5, 1, 256, 256)
    targets = torch.ones(5, 256, 256)
    assert pytest.approx(compute_loss(inputs, targets)["total"], abs=1e-4) == 0

    # Multiclass
    inputs = -100 * torch.ones(5, 3, 256, 256)
    targets = torch.zeros(5, 256, 256)
    inputs[:, 0, ...] = 100
    assert pytest.approx(compute_loss(inputs, targets)["total"], abs=1e-4) == 0


def test_composed_loss():
    compute_loss = ComposedLoss([FocalLoss(), TverskyLoss()])

    # Binary
    inputs = 10 * torch.ones(5, 1, 256, 256)
    targets = torch.ones(5, 256, 256)
    assert pytest.approx(compute_loss(inputs, targets)["total"], abs=1e-4) == 0

    # Multiclass
    inputs = -100 * torch.ones(5, 3, 256, 256)
    targets = torch.zeros(5, 256, 256)
    inputs[:, 0, ...] = 100
    assert pytest.approx(compute_loss(inputs, targets)["total"], abs=1e-4) == 0
