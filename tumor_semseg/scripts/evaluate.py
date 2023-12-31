"""
Evaluate model performance.
"""

# Third-Party
import hydra
import torch
import torch.ao.quantization as quantization
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from prettytable import PrettyTable
from torch import Tensor

# TumorSemSeg
from tumor_semseg.data.brain_mri_datamodule import BrainMRIDataModule
from tumor_semseg.loss.utils import compute_dice, compute_iou
from tumor_semseg.module.brain_mri_module import BrainMRIModule


def get_metrics_table_summary(metrics: dict[str, Tensor], dataset_id: str):
    # Reduce over batches
    metrics["mean_per_class_dice"] = metrics["dice"].mean(0)
    metrics["mean_per_class_iou"] = metrics["iou"].mean(0)

    # Reduce over classes
    metrics["mean_dice"] = metrics["mean_per_class_dice"].mean()
    metrics["mean_iou"] = metrics["mean_per_class_iou"].mean()

    table: PrettyTable = PrettyTable(title=f"Evaluation Summary ({dataset_id})", header=False)
    table.add_row(["Number of samples", metrics["dice"].size(0)])
    table.add_row(["Mean per class Dice", metrics["mean_per_class_dice"].tolist()])
    table.add_row(["Mean per class IoU", metrics["mean_per_class_iou"].tolist()])
    table.add_row(["Mean Dice", float(metrics["mean_dice"])])
    table.add_row(["Mean IoU", float(metrics["mean_iou"])])

    return table


def compute_metrics_from_output(output: list[Tensor, tuple[Tensor, Tensor]], dataset_id: str):
    metrics = {"dice": Tensor(), "iou": Tensor()}
    for y_hat, (_, y) in output:
        metrics["dice"] = torch.concat([metrics["dice"], compute_dice(y_hat, y)], dim=0)
        metrics["iou"] = torch.concat([metrics["iou"], compute_iou(y_hat, y)], dim=0)

    table = get_metrics_table_summary(metrics, dataset_id)

    return metrics, table


def compute_global_metrics(metrics_train, metrics_val):
    global_metrics = {}
    global_metrics["dice"] = torch.concat([metrics_train["dice"], metrics_val["dice"]], dim=0)
    global_metrics["iou"] = torch.concat([metrics_train["iou"], metrics_val["iou"]], dim=0)

    table = get_metrics_table_summary(global_metrics, "global")

    return global_metrics, table


@hydra.main(config_path="../configuration", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    assert cfg.checkpoint is not None, "checkpoint must be specified in config for evaluation to run"

    module: BrainMRIModule = BrainMRIModule.load_from_checkpoint(cfg.checkpoint)
    if hasattr(module, "qconfig"):
        cfg.trainer.precision = "32-true"
        cfg.trainer.accelerator = "cpu"  # Inference generally faster on CPU than CUDA device for int8 models
        module = module.get_quantized_model()

    cfg.datamodule.config.augment = False
    datamodule: BrainMRIDataModule = instantiate(cfg.datamodule)
    datamodule.setup("fit")

    trainer: Trainer = instantiate(cfg.trainer)
    trainer.logger.log_hyperparams(cfg)

    with torch.no_grad():
        output_train = trainer.predict(module, dataloaders=datamodule.train_dataloader())
        output_val = trainer.predict(module, dataloaders=datamodule.val_dataloader())

    metrics_train, table_train = compute_metrics_from_output(output_train, "train")
    metrics_val, table_val = compute_metrics_from_output(output_val, "val")
    _, table_global = compute_global_metrics(metrics_train, metrics_val)

    print(table_train)
    print(table_val)
    print(table_global)

    with open("evaluation.log", "w") as f:
        f.write("\n\n".join([str(table_train), str(table_val), str(table_global)]))


if __name__ == "__main__":
    main()
