"""
UNet module definition.
"""

# Standard
from dataclasses import dataclass
from typing import Optional

# Third-Party
import lightning as L
import torch
from torch import Tensor

# TumorSemSeg
from tumor_semseg.loss.semseg_losses import SemSegLoss
from tumor_semseg.optimize.optimizer import CustomOptimizer
from tumor_semseg.optimize.scheduler import CustomScheduler


@dataclass
class BrainMRIModuleConfig:
    model: L.LightningModule
    loss: SemSegLoss
    optimizer: CustomOptimizer
    scheduler: Optional[CustomScheduler] = None
    bin_det_threshold: float = 0.5
    example_input_array_shape: Optional[tuple[int, int, int, int]] = None

    def __post_init__(self):
        # NOTE: OmegaConfg does not currently support `tuple`
        if self.example_input_array_shape:
            self.example_input_array_shape = tuple(self.example_input_array_shape)


class BrainMRIModule(L.LightningModule):
    def __init__(self, config: BrainMRIModuleConfig):
        super().__init__()
        self.model = config.model
        self.loss_fn = config.loss
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.bin_det_threshold = config.bin_det_threshold
        if config.example_input_array_shape:
            self.example_input_array = torch.zeros(
                config.example_input_array_shape
            )  # Special Lightning attribute to compute I/O size of each layer for model summary

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.log_loss("train", loss)

        return {"loss": loss["total"], "pred": y_hat}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.log_loss("val", loss)

        return {"loss": loss["total"], "pred": y_hat}

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        if y_hat.size(1) == 1:
            return (y_hat.sigmoid() > self.bin_det_threshold).float()
        else:
            return y_hat.argmax(dim=1).float()

    def configure_optimizers(self):
        config = {"optimizer": self.optimizer.get_optimizer(self.model)}
        if self.scheduler:
            config["lr_scheduler"] = {"scheduler": self.scheduler.get_scheduler(config["optimizer"])}
            if self.scheduler.params:
                config["lr_scheduler"] |= self.scheduler.params
        return config

    def log_loss(self, stage: str, loss: dict[str, Tensor]):
        for loss_name, value in loss.items():
            self.log(
                f"{stage}_{loss_name}_loss",
                value,
                sync_dist=True,
                prog_bar=True if loss_name == "total" else False,
                on_step=False,
                on_epoch=True,
            )
