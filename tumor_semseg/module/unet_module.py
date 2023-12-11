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
    bin_det_threshold: float = 0.5
    image_size: Optional[tuple[int, int]] = None


class UNetModule(L.LightningModule):
    def __init__(self, config: UNetModuleConfig):
        super().__init__()
        if config.image_size:
            self.example_input_array = torch.Tensor(
                1, config.unet.in_channels, config.image_size[0], config.image_size[1]
            )  # Special Lightning attribute to compute I/O size of each layer for model summary
        self.model = UNet(config.unet)
        self.loss_fn = DiceCEEdgeLoss(config.loss)
        self.optimizer_config = config.optimizer
        self.bin_det_threshold = config.bin_det_threshold

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.predict_step(batch, 0)
        y_hat = self(x)
        losses = self.loss_fn(y_hat, y)

        self.log("train_total_loss", losses["total"], sync_dist=True, prog_bar=True)
        self.log("train_dice_loss", losses["dice"], sync_dist=True)
        self.log("train_ce_loss", losses["ce"], sync_dist=True)
        self.log("train_edge_loss", losses["edge"], sync_dist=True)

        return {"loss": losses["total"], "pred": y_hat}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        losses = self.loss_fn(y_hat, y)

        self.log("val_total_loss", losses["total"], sync_dist=True, prog_bar=True)
        self.log("val_dice_loss", losses["dice"], sync_dist=True)
        self.log("val_ce_loss", losses["ce"], sync_dist=True)
        self.log("val_edge_loss", losses["edge"], sync_dist=True)

        return {"loss": losses["total"], "pred": y_hat}

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        if y_hat.size(1) == 1:
            return (y_hat.sigmoid() > self.bin_det_threshold).float()
        else:
            return y_hat.argmax(dim=1).float()

    def configure_optimizers(self):
        optimizer = self.optimizer_config.optimizer
        return optimizer(self.model.parameters(), lr=self.optimizer_config.lr)
