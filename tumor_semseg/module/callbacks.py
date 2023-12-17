import aim
import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor
from torchvision.utils import make_grid

# Tumor SemSeg
from tumor_semseg.loss.utils import compute_iou


class PredVisualizationCallback(L.Callback):
    def __init__(self, log_every_n_batches: int, num_samples: int = 1):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.num_samples = num_samples

    @staticmethod
    def generate_pred_grid(x: Tensor, y: Tensor, y_hat: Tensor):
        y *= 255
        y_hat *= 255
        y = y.repeat_interleave(3, dim=1)
        y_hat = y_hat.repeat_interleave(3, dim=1)
        overlay = 0.7 * x + 0.3 * y

        grid = torch.cat((x, y, y_hat, overlay), dim=0)

        return make_grid(grid, nrow=4, padding=10, pad_value=255)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % min(self.log_every_n_batches, trainer.num_training_batches - 1) == 0:
            grid = PredVisualizationCallback.generate_pred_grid(batch[0], batch[1], outputs["pred"])
            trainer.logger.experiment.track(
                aim.Image(grid, caption="Image Training"), name="train", context={"context_key": "train_value"}
            )

    @rank_zero_only
    def on_val_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % min(self.log_every_n_batches, trainer.num_training_batches - 1) == 0:
            grid = PredVisualizationCallback.generate_pred_grid(batch[0], batch[1], outputs["pred"])
            trainer.logger.experiment.track(
                aim.Image(grid, caption="Image Validation"), name="val", context={"context_key": "val_value"}
            )


class ComputeIoUCallback(L.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _, y = batch
        y_hat = outputs["pred"]
        iou = compute_iou(y_hat, y)

        for class_idx, class_iou in enumerate(iou):
            pl_module.log(f"train_IoU_class_{class_idx}", class_iou, sync_dist=True)
        pl_module.log("train_mIoU", iou.mean(), sync_dist=True, prog_bar=True)

    def on_val_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _, y = batch
        y_hat = outputs["pred"]
        iou = compute_iou(y_hat, y)

        for class_idx, class_iou in enumerate(iou):
            pl_module.log(f"val_IoU_class_{class_idx}", class_iou, sync_dist=True)
        pl_module.log("val_mIoU", iou.mean(), sync_dist=True, prog_bar=True)
