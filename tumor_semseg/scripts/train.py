"""
This files defines the entry point for training.
"""

# Third-Party
import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig

# TumorSemSeg
from tumor_semseg.data.brain_mri_datamodule import BrainMRIDataModule
from tumor_semseg.module.brain_mri_module import BrainMRIModule
from tumor_semseg.module.quantization import auto_fuse_modules


@hydra.main(config_path="../configuration", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.seed:
        seed_everything(cfg.seed)

    brain_mri_module: BrainMRIModule = (
        BrainMRIModule(cfg.module.config)
        if cfg.checkpoint is None
        else BrainMRIModule.load_from_checkpoint(cfg.checkpoint)
    )
    brain_mri_datamodule: BrainMRIDataModule = instantiate(cfg.datamodule)

    if cfg.quantization.qat:
        cfg.trainer.precision = "32-true"  # Need to use float32 precision for QAT

    trainer: Trainer = instantiate(cfg.trainer)

    if cfg.find_initial_lr:
        tuner = Tuner(trainer)
        tuner.lr_find(brain_mri_module, brain_mri_datamodule)
        trainer = instantiate(cfg.trainer)  # Need to be reinstantiated when training on GPU

    if cfg.quantization.qat:
        brain_mri_module.eval()
        brain_mri_module.qconfig = torch.ao.quantization.get_default_qat_qconfig(cfg.quantization.qconfig)
        if cfg.quantization.fuse_modules:
            auto_fuse_modules(brain_mri_module)
        brain_mri_module = torch.ao.quantization.prepare_qat(brain_mri_module.train())

    trainer.logger.log_hyperparams(cfg)

    trainer.fit(brain_mri_module, datamodule=brain_mri_datamodule)


if __name__ == "__main__":
    main()
