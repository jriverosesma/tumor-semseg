# Standard
from typing import Optional

# Third-Party
import torch.nn.functional as F
from torch import Tensor


def reduce(x: Tensor, dims: Optional[list[int]] = None, reduction: str = "mean"):
    match reduction:
        case "mean":
            return x.mean(dims)
        case "sum":
            return x.sum(dims)
        case "none":
            return x
        case _:
            raise KeyError("reduction must be one of ['mean', 'sum', 'none']")


def one_hot_encode(targets: Tensor, n_classes: int):
    return targets if n_classes == 1 else F.one_hot(targets.long(), n_classes).permute(0, 3, 1, 2).float()


def compute_dice(inputs: Tensor, targets: Tensor, smooth: float = 1e-6):
    n_batches = inputs.size(0)
    n_classes = inputs.size(1)
    input_probs = inputs.sigmoid() if n_classes == 1 else inputs.softmax(dim=1)
    input_probs = input_probs.view(n_batches, n_classes, -1)

    targets_one_hot = one_hot_encode(targets, n_classes).view(n_batches, n_classes, -1)

    intersection = (input_probs * targets_one_hot).sum(-1)
    sum_areas = (input_probs + targets_one_hot).sum(-1)

    return (2.0 * intersection + smooth) / (sum_areas + smooth)


def compute_iou(inputs: Tensor, targets: Tensor, smooth: float = 1e-10):
    dice = compute_dice(inputs, targets, smooth)
    return dice / (2 - dice)


def compute_tversky(inputs: Tensor, targets: Tensor, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
    n_batches = inputs.size(0)
    n_classes = inputs.size(1)
    input_probs = inputs.sigmoid() if n_classes == 1 else inputs.softmax(dim=1)
    input_probs = input_probs.view(n_batches, n_classes, -1)

    targets_one_hot = one_hot_encode(targets, n_classes)
    targets_one_hot = targets_one_hot.view(n_batches, n_classes, -1)

    tp = (input_probs * targets_one_hot).sum(dim=-1)
    fp = ((1 - targets_one_hot) * input_probs).sum(dim=-1)
    fn = (targets_one_hot * (1 - input_probs)).sum(dim=-1)

    return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
