"""
UNet module definition.
"""

# Standard
from dataclasses import dataclass
from typing import Optional

# Third-Party
import lightning as L
import torch
import torch.ao.quantization as quantization
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn

# TumorSemSeg
from tumor_semseg.architecture.utils import auto_fuse_modules
from tumor_semseg.loss.semseg_losses import SemSegLoss
from tumor_semseg.module.utils import ExportableModel
from tumor_semseg.optimize.optimizer import CustomOptimizer
from tumor_semseg.optimize.scheduler import CustomScheduler


@dataclass
class QuantizationConfig:
    qconfig: str = "x86"
    auto_fuse_modules: bool = True


@dataclass
class BrainMRIModuleConfig:
    model: nn.Module
    loss: SemSegLoss
    optimizer: CustomOptimizer
    scheduler: Optional[CustomScheduler] = None
    bin_det_threshold: float = 0.5
    example_input_array_shape: Optional[tuple[int, int, int, int]] = None
    qat: Optional[QuantizationConfig] = None

    def __post_init__(self):
        # NOTE: OmegaConf does not currently support `tuple`
        if self.example_input_array_shape:
            self.example_input_array_shape = tuple(self.example_input_array_shape)


class BrainMRIModule(L.LightningModule):
    def __init__(self, config: BrainMRIModuleConfig | DictConfig):
        super().__init__()
        # NOTE: Hyper-parameters not saved if config is instantiated when passed as arg
        self.save_hyperparameters()

        if isinstance(config, DictConfig):
            config: BrainMRIModuleConfig = instantiate(config)

        self.model = config.model
        self.loss_fn = config.loss
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.bin_det_threshold = config.bin_det_threshold
        self.example_input_array = torch.ones(
            config.example_input_array_shape
        )  # Special Lightning attribute to compute I/O size of each layer for model summary
        self.lr = None  # Special Lightning attribute used for initial LR tuning only

        if config.qat:
            self.quant = quantization.QuantStub()
            self.dequant = quantization.DeQuantStub()
            self.eval()
            self.qconfig = quantization.get_default_qat_qconfig(config.qat.qconfig)
            if config.qat.auto_fuse_modules:
                auto_fuse_modules(self.model)
            quantization.prepare_qat(self.train(), inplace=True)

    def forward(self, inputs):
        if hasattr(self, "qconfig"):
            return self.dequant(self.model(self.quant(inputs)))
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.log_loss("train", loss)

        return {"loss": loss["total"], "preds": y_hat}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.log_loss("val", loss)

        return {"loss": loss["total"], "preds": y_hat}

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        if y_hat.size(1) == 1:
            return (y_hat.sigmoid() > self.bin_det_threshold).float(), batch
        else:
            return y_hat.argmax(dim=1).float(), batch

    def configure_optimizers(self):
        # For initial LR tuning
        if self.lr is not None:
            self.optimizer.args["lr"] = self.lr
        config = {"optimizer": self.optimizer.get_optimizer(self)}
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

    def get_exportable_model(self):
        return ExportableModel(self)
