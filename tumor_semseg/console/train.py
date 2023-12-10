"""
This files defines the entry point for training.
"""

from pathlib import Path

import lightning as L
from lightning.pytorch import seed_everything

# Tumor SemSeg
from tumor_semseg.architecture.unet import UNetConfig
from tumor_semseg.dataset.brain_mri import BrainMRIDataModule, BrainMRIDataModuleConfig
from tumor_semseg.loss.dice_ce_edge_loss import DiceCEEdgeLossConfig
from tumor_semseg.module.unet_module import OptimizerConfig, UNetModule, UNetModuleConfig

SEED = 2023


def main():
    # Set deterministic behavior
    seed_everything(SEED, workers=True)

    # Define problem parameters
    dataset_dirpath = Path("data")
    n_classes = 1

    # Instantiate configurations module configurations
    unet_config = UNetConfig(n_classes)
    loss_config = DiceCEEdgeLossConfig(n_classes)
    optimizer_config = OptimizerConfig(lr=1e-3)
    unet_module_config = UNetModuleConfig(unet_config, loss_config, optimizer_config)
    brain_mri_data_module_config = BrainMRIDataModuleConfig(dataset_dirpath, seed=SEED, batch_size=32, augment=False)

    # Instantiate Lightning modules
    model = UNetModule(unet_module_config)
    taxi_datamodule = BrainMRIDataModule(brain_mri_data_module_config)

    # Instantiate trainer
    trainer = L.Trainer(
        devices=1,
        accelerator="gpu",
        precision="16-mixed",
        # precision="bf16-mixed", # Preferred over "16-mixed" if supported for the GPU in use
        # sync_batchnorm=True, # Turn on when using multi-GPU and DDP
        deterministic=False,
        max_epochs=500,
        # profiler="simple",
    )

    # Train
    taxi_datamodule.prepare_data()
    trainer.fit(model, datamodule=taxi_datamodule)


if __name__ == "__main__":
    main()
