"""
UNet module definition.
"""

from dataclasses import dataclass
from typing import Optional

import lightning as L
import torch
from torch import Tensor

# Tumor SemSeg
from tumor_semseg.loss.semseg_losses import SemSegLoss
from tumor_semseg.optimizer.base_optimizer import BaseOptimizer, BaseScheduler


@dataclass
class BrainMRIModuleConfig:
    model: L.LightningModule
    loss: SemSegLoss
    optimizer: BaseOptimizer
    scheduler: BaseScheduler
    bin_det_threshold: float = 0.5
    example_input_array_shape: tuple[int, int, int, int] = None


class BrainMRIModule(L.LightningModule):
    def __init__(self, config: BrainMRIModuleConfig):
        super().__init__()
        self.model = config.model
        self.loss_fn = config.loss
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.bin_det_threshold = config.bin_det_threshold
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
        optimizer = self.optimizer.get_optimizer(self.model)
        scheduler = self.scheduler.get_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # "monitor": "metric_to_track",
                # "frequency": "indicates how often the metric is updated"
                # # If "monitor" references validation metrics, then "frequency" should be set to a
                # # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def log_loss(self, stage: str, loss: dict[str, Tensor]):
        for loss_name, value in loss.items():
            self.log(
                f"{stage}_{loss_name}_loss", value, sync_dist=True, prog_bar=True if loss_name == "total" else False
            )
