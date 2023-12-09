"""
UNet module definition.
"""

from dataclasses import dataclass
from typing import Optional

import lightning as L
import torch

# Tumor SemSeg
from tumor_semseg.architecture.unet import UNet, UNetConfig
from tumor_semseg.loss.dice_ce_edge_loss import DiceCEEdgeLoss, DiceCEEdgeLossConfig


@dataclass
class OptimizerConfig:
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    lr: Optional[float] = None


@dataclass
class UNetModuleConfig:
    unet: UNetConfig
    loss: DiceCEEdgeLossConfig
    optimizer: OptimizerConfig


class UNetModule(L.LightningModule):
    def __init__(self, config: UNetModuleConfig):
        super().__init__()
        self.model = UNet(config.unet)
        self.loss_fn = DiceCEEdgeLoss(config.loss)
        self.optimizer_config = config.optimizer

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        losses = self.loss_fn(output, target)

        self.log("train_total_loss", losses["total"])
        self.log("train_dice_loss", losses["dice"])
        self.log("train_ce_loss", losses["ce"])
        self.log("train_edge_loss", losses["edge"])

        return losses["total"]

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        losses = self.loss_fn(output, target)

        self.log("val_total_loss", losses["total"])
        self.log("val_dice_loss", losses["dice"])
        self.log("val_ce_loss", losses["ce"])
        self.log("val_edge_loss", losses["edge"])

        return losses["total"]

    def configure_optimizers(self):
        optimizer = self.optimizer_config.optimizer
        return optimizer(self.model.parameters(), lr=self.optimizer_config.lr)
