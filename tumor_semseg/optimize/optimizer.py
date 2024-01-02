"""
Definition of custom optimizers.
"""

# Standard
from functools import partial
from typing import Any, Type

# Third-Party
import torch
from torch.optim import SGD, Adam, Adamax, Optimizer, RMSprop


class CustomOptimizer:
    def __init__(self, optimizer_class: Optimizer, optimizer_args: dict[str, Any]):
        self.optimizer_class: Type[Optimizer] = optimizer_class
        self.optimizer_args: dict[str, Any] = optimizer_args

    def get_optimizer(self, model: torch.nn.Module):
        return self.optimizer_class(model.parameters(), **self.optimizer_args)


adam: CustomOptimizer = partial(CustomOptimizer, Adam, {"lr": 1e-3, "weight_decay": 1e-4})
adamax: CustomOptimizer = partial(CustomOptimizer, Adamax, {"lr": 1e-3, "weight_decay": 1e-4})
sgd: CustomOptimizer = partial(CustomOptimizer, SGD, {"lr": 1e-3, "weight_decay": 1e-4})
rmsprop: CustomOptimizer = partial(CustomOptimizer, RMSprop, {"lr": 1e-3, "weight_decay": 1e-4})
