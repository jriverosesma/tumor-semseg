import torch.nn.functional as F
from torch import Tensor


def one_hot_encode(targets: Tensor, n_classes: int):
    return targets if n_classes == 1 else F.one_hot(targets.long(), n_classes).squeeze(1).permute(0, 3, 1, 2).float()


def compute_dice(inputs: Tensor, targets: Tensor, smooth: float = 1e-6):
    n_batches = inputs.size(0)
    n_classes = inputs.size(1)
    inputs = inputs.sigmoid() if n_classes == 1 else inputs.softmax(dim=1)
    inputs = inputs.view(n_batches, n_classes, -1)

    targets_one_hot = one_hot_encode(targets, n_classes)
    targets_one_hot = targets_one_hot.view(n_batches, n_classes, -1)

    intersection = (inputs * targets_one_hot).sum(-1)
    sum_areas = inputs.sum(-1) + targets_one_hot.sum(-1)

    return (2.0 * intersection + smooth) / (sum_areas + smooth)


def compute_iou(inputs: Tensor, targets: Tensor, smooth: float = 1e-6):
    dice = compute_dice(inputs, targets, smooth)
    return dice / (2 - dice)
