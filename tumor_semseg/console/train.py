"""
This files defines the entry point for training.
"""

from pathlib import Path

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_summary import ModelSummary

# Tumor SemSeg
from tumor_semseg.architecture.unet import UNetConfig
from tumor_semseg.dataset.brain_mri import BrainMRIDataModule, BrainMRIDataModuleConfig
from tumor_semseg.loss.dice_ce_edge_loss import DiceCEEdgeLossConfig
from tumor_semseg.module.callbacks import ComputeIoUCallback, PredVisualizationCallback
from tumor_semseg.module.unet_module import OptimizerConfig, UNetModule, UNetModuleConfig

SEED = 2023


def main():
    # Set deterministic behavior
    seed_everything(SEED, workers=True)

    # Define problem parameters
    dataset_dirpath = Path("data")
    image_size = (256, 256)
    n_classes = 1
    # Logs
    log_every_n_batches = 16
    n_pred_samples = 4

    # Instantiate configurations
    unet_config = UNetConfig(n_classes)
    loss_config = DiceCEEdgeLossConfig()
    optimizer_config = OptimizerConfig(lr=1e-3)
    unet_module_config = UNetModuleConfig(unet_config, loss_config, optimizer_config, image_size=image_size)
    brain_mri_data_module_config = BrainMRIDataModuleConfig(
        dataset_dirpath, seed=SEED, batch_size=128, augment=True, num_workers=6, image_size=image_size
    )

    # Instantiate Lightning modules
    model = UNetModule(unet_module_config)
    brain_mri_datamodule = BrainMRIDataModule(brain_mri_data_module_config)

    # Instantiate trainer
    brain_mri_datamodule.setup("fit")
    trainer = L.Trainer(
        accelerator="cpu",
        # accelerator="gpu",
        # precision="16-mixed",
        # precision="bf16-mixed",  # Preferred over "16-mixed" if supported for the GPU in use
        sync_batchnorm=True,  # Turn on when using multi-GPU and DDP
        deterministic=False,
        max_epochs=500,
        # profiler="simple",
        callbacks=[
            EarlyStopping(monitor="val_total_loss", mode="min", min_delta=1e-4),
            ModelSummary(max_depth=-1),
            DeviceStatsMonitor(cpu_stats=True),
            ComputeIoUCallback(),
            PredVisualizationCallback(log_every_n_batches=log_every_n_batches, num_samples=n_pred_samples),
        ],
    )

    # Train
    trainer.fit(model, datamodule=brain_mri_datamodule)


if __name__ == "__main__":
    main()
