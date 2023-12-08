import torch
import torch.nn.functional as F

# Tumor SemSeg
from tumor_semseg.loss.dice_ce_edge_loss import (
    CrossEntropyLoss,
    DiceCEEdgeLoss,
    DiceCEEdgeLossConfig,
    DiceLoss,
    EdgeLoss,
)


def test_dice_loss():
    inputs = torch.tensor([0.5, 0.8, 0.1])
    targets = torch.tensor([1.0, 1.0, 0.0])
    smooth = 1e-6
    dice_loss = DiceLoss(smooth)

    loss = dice_loss(inputs, targets)
    expected_loss = 1 - (2.0 * 1.3 + smooth) / (1.4 + 2.0 + smooth)

    assert torch.isclose(loss, torch.tensor(expected_loss))


def test_cross_entropy_loss():
    inputs = torch.randn(3, 5, requires_grad=True)
    targets = torch.empty(3, dtype=torch.long).random_(5)
    weight = torch.tensor([0.2, 0.3, 0.1, 0.4, 0.1])
    ce_loss = CrossEntropyLoss(weight)

    loss = ce_loss(inputs, targets)
    expected_loss = F.cross_entropy(inputs, targets, weight)

    assert torch.isclose(loss, expected_loss)


def test_edge_loss():
    inputs = torch.rand(1, 1, 10, 10)
    targets = torch.rand(1, 1, 10, 10)
    edge_loss = EdgeLoss()

    loss = edge_loss(inputs, targets)

    assert loss is not None


def test_dice_ce_edge_loss():
    inputs = torch.rand(1, 10, 10)
    targets = torch.rand(1, 10, 10)
    config = DiceCEEdgeLossConfig(n_classes=10)
    combined_loss = DiceCEEdgeLoss(config)

    loss = combined_loss(inputs, targets)

    assert loss is not None, "DiceCEEdgeLoss did not compute a loss value"
