"""
Base optimizer and scheduler classes.
"""

from dataclasses import dataclass, field
from typing import Any, Type

import torch


@dataclass
class BaseOptimizer:
    optimizer_class: Type[torch.optim.Optimizer]
    optimizer_params: dict[str, Any] = field(default_factory=lambda: {})

    def get_optimizer(self, model: torch.nn.Module):
        return self.optimizer_class(model.parameters(), **self.optimizer_params)


@dataclass
class BaseScheduler:
    scheduler_class: Type[torch.optim.lr_scheduler.LRScheduler]
    scheduler_params: dict[str, Any] = field(default_factory=lambda: {})

    def get_scheduler(self, optimizer):
        return self.scheduler_class(optimizer, **self.scheduler_params)
