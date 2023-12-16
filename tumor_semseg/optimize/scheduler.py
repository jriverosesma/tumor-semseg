"""
Definition of custom schedulers.
"""

from functools import partial
from typing import Any, Optional, Type

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


class CustomScheduler:
    def __init__(self, scheduler_class: LRScheduler, args: dict[str, Any]):
        self.scheduler_class: Type[LRScheduler] = scheduler_class
        self.args: dict[str, Any] = args
        self.params: Optional[dict[str, Any]] = None

    def get_scheduler(self, optimizer: Optimizer):
        return self.scheduler_class(optimizer, **self.args)


lambda_lr: CustomScheduler = partial(
    CustomScheduler, LambdaLR, {"lr_lambda": [lambda epoch: epoch // 30, lambda epoch: 0.95**epoch]}
)
