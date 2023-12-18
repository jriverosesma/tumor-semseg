"""
Definition of custom optimizers.
"""

# Standard
from functools import partial
from typing import Any, Type

# Third-Party
import torch
from torch.optim import Adamax, Optimizer


class CustomOptimizer:
    def __init__(self, optimizer_class: Optimizer, args: dict[str, Any]):
        self.optimizer_class: Type[Optimizer] = optimizer_class
        self.args: dict[str, Any] = args

    def get_optimizer(self, model: torch.nn.Module):
        return self.optimizer_class(model.parameters(), **self.args)


adam: CustomOptimizer = partial(CustomOptimizer, Adamax, {"lr": 1e-3, "weight_decay": 1e-4})
adamax: CustomOptimizer = partial(CustomOptimizer, Adamax, {"lr": 1e-3, "weight_decay": 1e-4})
