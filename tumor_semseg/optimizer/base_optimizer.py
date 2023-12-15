"""
Base optimizer and scheduler classes.
"""

from dataclasses import dataclass, field
from typing import Any, Type

import torch


@dataclass
class BaseOptimizer:
    optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adamax
    params: dict[str, Any] = field(default_factory=lambda: {"lr": 1e-3})

    def get_optimizer(self, model: torch.nn.Module):
        return self.optimizer_class(model.parameters(), **self.params)


@dataclass
class BaseScheduler:
    scheduler_class: Type[torch.optim.lr_scheduler.LRScheduler] = torch.optim.lr_scheduler.LambdaLR
    params: dict[str, Any] = field(
        default_factory=lambda: {"lr_lambda": [lambda epoch: epoch // 30, lambda epoch: 0.95**epoch]}
    )

    def get_scheduler(self, optimizer: torch.optim.Optimizer):
        return self.scheduler_class(optimizer, **self.params)
