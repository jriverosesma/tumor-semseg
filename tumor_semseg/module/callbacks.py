# Third-Party
import aim
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor

# TumorSemSeg
from tumor_semseg.loss.utils import compute_iou


class PredVisualizationCallback(L.Callback):
    def __init__(self, log_every_n_batches: int, n_samples: int = 5):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.n_samples = n_samples

    @staticmethod
    def generate_pred_visualization(x: Tensor, y: Tensor, y_hat: Tensor, bin_det_threshold: float):
        y_hat = y_hat.sigmoid().detach()
        y_hat_t = torch.where(y_hat > bin_det_threshold, 1.0, 0.0)
        iou = float(compute_iou(y_hat_t.unsqueeze(0), y))

        y_hat_np = y_hat.float().cpu().numpy()
        y_hat_t_np = y_hat_t.float().cpu().numpy()
        x_np = x.float().cpu().numpy()
        y_np = y.float().cpu().numpy()

        norm_image = (x_np - x_np.min()) / (x_np.max() - x_np.min())
        norm_image = 255 * norm_image.transpose(1, 2, 0)
        mask = 255 * y_np
        pred = y_hat_np.squeeze(0)
        tpred = y_hat_t_np.squeeze(0)
        pred *= 255

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        fig.suptitle(f"IoU: {np.round(iou, 2)}")
        ax[0, 0].imshow(norm_image.astype(np.uint8))
        ax[0, 0].set_title("image")
        ax[0, 1].imshow(mask.astype(np.uint8))
        ax[0, 1].set_title("mask")
        ax[1, 0].imshow(pred.astype(np.uint8))
        ax[1, 0].set_title("prediction")
        ax[1, 1].imshow(tpred.astype(np.uint8))
        ax[1, 1].set_title("thresholded prediction")

        return fig

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % min(self.log_every_n_batches, trainer.num_training_batches) == 0:
            for i in range(min(self.n_samples, batch[0].size(0))):
                fig = PredVisualizationCallback.generate_pred_visualization(
                    batch[0][i], batch[1][i], outputs["preds"][i], pl_module.bin_det_threshold
                )
                trainer.logger.experiment.track(
                    aim.Image(fig), name="image", epoch=trainer.current_epoch, context={"subset": "train"}
                )
                plt.close()

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % min(self.log_every_n_batches, trainer.num_val_batches[0]) == 0:
            for i in range(min(self.n_samples, batch[0].size(0))):
                fig = PredVisualizationCallback.generate_pred_visualization(
                    batch[0][i], batch[1][i], outputs["preds"][i], pl_module.bin_det_threshold
                )
                trainer.logger.experiment.track(
                    aim.Image(fig), name="image", epoch=trainer.current_epoch, context={"subset": "val"}
                )
                plt.close()


class ComputeIoUCallback(L.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _, y = batch
        y_hat = torch.where(outputs["preds"].sigmoid() > pl_module.bin_det_threshold, 1.0, 0.0)
        iou = compute_iou(y_hat, y).mean(0)  # Average over batches

        for class_idx, class_iou in enumerate(iou):
            pl_module.log(f"train_IoU_class_{class_idx}", class_iou, sync_dist=True)
        pl_module.log("train_mIoU", iou.mean(), sync_dist=True, prog_bar=True, on_epoch=True, on_step=False)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _, y = batch
        y_hat = torch.where(outputs["preds"].sigmoid() > pl_module.bin_det_threshold, 1.0, 0.0)
        iou = compute_iou(y_hat, y).mean(0)  # Average over batches

        for class_idx, class_iou in enumerate(iou):
            pl_module.log(f"val_IoU_class_{class_idx}", class_iou, sync_dist=True)
        pl_module.log("val_mIoU", iou.mean(), sync_dist=True, prog_bar=True, on_epoch=True, on_step=False)
