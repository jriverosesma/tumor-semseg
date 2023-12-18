# Third-Party
import aim
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor

# TumorSemSeg
from tumor_semseg.loss.utils import compute_iou


class PredVisualizationCallback(L.Callback):
    def __init__(self, log_every_n_batches: int, num_samples: int = 1):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.num_samples = num_samples

    @staticmethod
    def generate_pred_visualization(x: Tensor, y: Tensor, y_hat: Tensor, bin_det_threshold: float):
        norm_image = (x - x.min()) / (x.max() - x.min())
        norm_image = 255 * norm_image.permute(1, 2, 0)
        mask = 255 * y
        pred = y_hat.float().detach().numpy().squeeze(0)
        tpred = np.zeros(pred.shape)
        tpred[pred > bin_det_threshold] = 1.0

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        ax[0, 0].imshow(norm_image)
        ax[0, 0].set_title("image")
        ax[0, 1].imshow(mask)
        ax[0, 1].set_title("mask")
        ax[1, 0].imshow(pred)
        ax[1, 0].set_title("prediction")
        ax[1, 1].imshow(tpred)
        ax[1, 1].set_title("thresholded prediction")

        return fig

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % min(self.log_every_n_batches, trainer.num_training_batches) == 0:
            for i in range(min(self.num_samples, batch[0].size(0))):
                fig = PredVisualizationCallback.generate_pred_visualization(
                    batch[0][i], batch[1][i], outputs["pred"][i], pl_module.bin_det_threshold
                )
                trainer.logger.experiment.track(
                    aim.Image(fig, caption="Image Training"), name="train", context={"context_key": "train_value"}
                )

    @rank_zero_only
    def on_val_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % min(self.log_every_n_batches, trainer.num_val_batches) == 0:
            for i in range(min(self.num_samples, batch[0].size(0))):
                fig = PredVisualizationCallback.generate_pred_visualization(
                    batch[0][i], batch[1][i], outputs["pred"][i], pl_module.bin_det_threshold
                )
                trainer.logger.experiment.track(
                    aim.Image(fig, caption="Image Validation"), name="train", context={"context_key": "val_value"}
                )


class ComputeIoUCallback(L.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _, y = batch
        y_hat = outputs["pred"]
        iou = compute_iou(y_hat, y).mean(0)  # Average over batches

        for class_idx, class_iou in enumerate(iou):
            pl_module.log(f"train_IoU_class_{class_idx}", class_iou, sync_dist=True)
        pl_module.log("train_mIoU", iou.mean(), sync_dist=True, prog_bar=True)

    def on_val_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _, y = batch
        y_hat = outputs["pred"]
        iou = compute_iou(y_hat, y).mean(0)  # Average over batches

        for class_idx, class_iou in enumerate(iou):
            pl_module.log(f"val_IoU_class_{class_idx}", class_iou, sync_dist=True)
        pl_module.log("val_mIoU", iou.mean(), sync_dist=True, prog_bar=True)
