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
    def __init__(self, optimizer_class: Optimizer, args: dict[str, Any]):
        self.optimizer_class: Type[Optimizer] = optimizer_class
        self.args: dict[str, Any] = args

    def get_optimizer(self, model: torch.nn.Module):
        return self.optimizer_class(model.parameters(), **self.args)


adam: CustomOptimizer = partial(CustomOptimizer, Adam)
adamax: CustomOptimizer = partial(CustomOptimizer, Adamax)
sgd: CustomOptimizer = partial(CustomOptimizer, SGD)
rmsprop: CustomOptimizer = partial(CustomOptimizer, RMSprop)
