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
    dataset_dir = Path("data/kaggle_3m")
    n_classes = 2

    # Instantiate configurations module configurations
    unet_config = UNetConfig(n_classes)
    loss_config = DiceCEEdgeLossConfig(n_classes)
    optimizer_config = OptimizerConfig(lr=1e-3)
    unet_module_config = UNetModuleConfig(unet_config, loss_config, optimizer_config)
    brain_mri_data_module_config = BrainMRIDataModuleConfig(dataset_dir, seed=SEED)

    # Instantiate Lightning modules
    model = UNetModule(unet_module_config)
    taxi_datamodule = BrainMRIDataModule(brain_mri_data_module_config)

    # Instantiate trainer
    trainer = L.Trainer(deterministic=False)

    # Train
    taxi_datamodule.prepare_data()

    # BUG: We should not need to call this.
    taxi_datamodule.setup("fit")
    taxi_datamodule.setup("validate")

    trainer.fit(model, datamodule=taxi_datamodule)


if __name__ == "__main__":
    main()
