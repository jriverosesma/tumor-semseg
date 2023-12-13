"""
This files defines the entry point for training.
"""

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_summary import ModelSummary

# Tumor SemSeg
from tumor_semseg.architecture.deeplab import DeepLabv3, DeepLabv3Config
from tumor_semseg.architecture.unet import UNet, UNetConfig
from tumor_semseg.data.brain_mri_datamodule import BrainMRIDataModule, BrainMRIDataModuleConfig
from tumor_semseg.loss.semseg_losses import DiceFocalEdgeLoss, DiceFocalEdgeLossConfig
from tumor_semseg.module.brain_mri_module import BrainMRIModule, BrainMRIModuleConfig
from tumor_semseg.module.callbacks import ComputeIoUCallback, PredVisualizationCallback
from tumor_semseg.optimizer.base_optimizer import BaseOptimizer, BaseScheduler

# SEED = 2023
SEED = None
MODEL_NAMES = ["unet", "deeplab"]


def main(dataset_dirpath: Path, model_name: str, image_size: tuple[int, int]):
    # Set deterministic behavior
    # seed_everything(SEED, workers=True)

    # Define problem parameters
    n_classes = 1

    # Logs
    log_every_n_batches = 16

    # Instantiate configurations
    assert model_name in MODEL_NAMES, f"model_name should be one of {MODEL_NAMES}"
    match model_name:
        case "unet":
            model_config = UNetConfig(n_classes)
            model = UNet(model_config)
        case "deeplab":
            model_config = DeepLabv3Config(n_classes)
            model = DeepLabv3(model_config)

    loss = DiceFocalEdgeLoss(DiceFocalEdgeLossConfig())
    optimizer = BaseOptimizer(torch.optim.Adamax, {"lr": 1e-3})
    # TODO: better tune this
    scheduler = BaseScheduler(
        torch.optim.lr_scheduler.LambdaLR, {"lr_lambda": lambda epoch: max(0.99**epoch, 1e-2 / 1e-3)}
    )
    brain_mri_module_config = BrainMRIModuleConfig(
        model,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        example_input_array_shape=(1, model_config.in_channels, image_size[0], image_size[1]),
    )
    brain_mri_datamodule_config = BrainMRIDataModuleConfig(
        dataset_dirpath, seed=SEED, batch_size=32, augment=True, num_workers=6, image_size=image_size
    )

    # Instantiate Lightning modules
    brain_mri_model = BrainMRIModule(brain_mri_module_config)
    brain_mri_datamodule = BrainMRIDataModule(brain_mri_datamodule_config)

    # Instantiate trainer
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # precision="16-mixed",
        # precision="bf16-mixed",  # Preferred over "16-mixed" if supported for the GPU in use
        sync_batchnorm=True,  # Turn on when using multi-GPU and DDP
        deterministic=False,
        max_epochs=500,
        # profiler="simple",
        callbacks=[
            # EarlyStopping(monitor="val_total_loss", mode="min", min_delta=1e-4),
            ModelSummary(max_depth=-1),
            DeviceStatsMonitor(cpu_stats=True),
            ComputeIoUCallback(),
            PredVisualizationCallback(log_every_n_batches=log_every_n_batches),
        ],
    )

    # Train
    trainer.fit(brain_mri_model, datamodule=brain_mri_datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training a model on the Brain MRI dataset.")

    parser.add_argument("--dataset_path", type=Path, default="data/kaggle_3m", help="Path to the dataset directory.")
    parser.add_argument("--model_name", type=str, default="unet", help="Name of the model to use.")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the input images (assumes square images).")

    args = parser.parse_args()

    main(args.dataset_path, args.model_name, (args.image_size, args.image_size))
