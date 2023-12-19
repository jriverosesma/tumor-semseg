"""
This files defines the entry point for training.
"""

# Third-Party
import hydra
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig

# TumorSemSeg
from tumor_semseg.data.brain_mri_datamodule import BrainMRIDataModule
from tumor_semseg.module.brain_mri_module import BrainMRIModule


@hydra.main(config_path="../configuration", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.seed:
        seed_everything(cfg.seed)

    brain_mri_model: BrainMRIModule = (
        instantiate(cfg.module) if cfg.checkpoint is None else BrainMRIModule.load_from_checkpoint(cfg.checkpoint)
    )
    brain_mri_datamodule: BrainMRIDataModule = instantiate(cfg.datamodule)
    trainer: Trainer = instantiate(cfg.trainer)

    trainer.logger.log_hyperparams(cfg)

    if cfg.find_initial_lr:
        tuner = Tuner(trainer)
        tuner.lr_find(brain_mri_model, brain_mri_datamodule)

    trainer.fit(brain_mri_model, datamodule=brain_mri_datamodule)


if __name__ == "__main__":
    main()
