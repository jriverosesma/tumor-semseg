"""
This files defines the entry point for training.
"""

# Standard
import warnings

# Third-Party
import hydra
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig

# TumorSemSeg
from tumor_semseg.data.brain_mri_datamodule import BrainMRIDataModule
from tumor_semseg.module.brain_mri_module import BrainMRIModule
from tumor_semseg.module.callbacks import CustomModelPruning


@hydra.main(config_path="../configuration", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> float:
    if cfg.seed:
        seed_everything(cfg.seed)

    module: BrainMRIModule = (
        BrainMRIModule(cfg.module.config)
        if cfg.checkpoint is None
        else BrainMRIModule.load_from_checkpoint(cfg.checkpoint)
    )
    datamodule: BrainMRIDataModule = instantiate(cfg.datamodule)

    if cfg.module.config.qat:
        cfg.trainer.precision = "32-true"  # Need to use float32 precision for QAT
        model_pruning_class_path = f"{CustomModelPruning.__module__}.{CustomModelPruning.__name__}"
        for i, callback in enumerate(cfg.trainer.callbacks):
            if callback["_target_"] == model_pruning_class_path:
                cfg.trainer.callbacks.pop(i)
                warnings.warn(
                    "Pruning not compatible with QAT (checkpoint loading will not work). Turning off pruning.",
                    UserWarning,
                )
                break

    trainer: Trainer = instantiate(cfg.trainer)

    if cfg.find_initial_lr:
        tuner = Tuner(trainer)
        tuner.lr_find(module, datamodule)
        trainer = instantiate(cfg.trainer)  # Need to be reinstantiated when training on GPU

    trainer.logger.log_hyperparams(cfg)

    trainer.fit(module, datamodule=datamodule)

    # Return best model score
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            return float(callback.best_model_score)


if __name__ == "__main__":
    main()
