"""
Definition of custom schedulers.
"""

# Standard
from functools import partial
from typing import Any, Optional, Type

# Third-Party
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, ReduceLROnPlateau


class CustomScheduler:
    def __init__(self, scheduler_class: LRScheduler, args: dict[str, Any], params: Optional[dict[str, Any]] = None):
        self.scheduler_class: Type[LRScheduler] = scheduler_class
        self.args: dict[str, Any] = args
        self.params: Optional[dict[str, Any]] = params

    def get_scheduler(self, optimizer: Optimizer):
        return self.scheduler_class(optimizer, **self.args)

    def get_params(self):
        return self.params


reduce_lr_on_plateau: CustomScheduler = partial(
    CustomScheduler,
    ReduceLROnPlateau,
    {"mode": "max", "factor": 0.2, "patience": 2, "threshold": 5e-4},
    {"monitor": "val_mIoU"},
)
lambda_lr: CustomScheduler = partial(CustomScheduler, LambdaLR, {"lr_lambda": lambda epoch: 0.85**epoch})
