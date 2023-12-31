"""
Entry point to run inferences on given images.
"""

# Standard
import glob
from pathlib import Path

# Third-Party
import albumentations as A
import cv2
import hydra
import numpy as np
import torch
import torch.ao.quantization as quantization
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# TumorSemSeg
from tumor_semseg.module.brain_mri_module import BrainMRIModule


class BrainMRIInferenceDataset(Dataset):
    def __init__(self, dataset_dirpath: Path, grayscale: bool, image_size: tuple[int, int]):
        self.images = BrainMRIInferenceDataset._get_image_paths(dataset_dirpath)
        self.grayscale = grayscale
        self.image_size = image_size
        self.transform = A.Compose(
            [A.Normalize(p=1.0, mean=0.5, std=0.5) if self.grayscale else A.Normalize(p=1.0), ToTensorV2()]
        )

    @staticmethod
    def _get_image_paths(dataset_dirpath: Path):
        assert dataset_dirpath.exists()

        if dataset_dirpath.is_dir():
            images = []
            for ext in ["png", "jpg", "jpeg", "tif"]:
                images += glob.glob(str(dataset_dirpath / f"**/*.{ext}"))
            images = [image for image in images if "mask" not in image]
            return images
        else:
            return str(dataset_dirpath)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = (
            Image.open(self.images[idx])
            .convert("L" if self.grayscale else "RGB")
            .resize(self.image_size, resample=Resampling.BILINEAR)
        )
        image = np.array(image)
        input_image = self.transform(image=image.astype(np.float32))["image"]

        return input_image.float(), image, self.images[idx]


@hydra.main(config_path="../configuration", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    assert cfg.checkpoint is not None, "checkpoint must be specified in config for inference to run"

    output_path = Path("preds")
    output_path.mkdir(exist_ok=True)

    match cfg.in_channels:
        case 3:
            grayscale = False
        case 1:
            grayscale = True
        case _:
            raise KeyError("in_channels must be one 3 (RGB) or 1 (L)")

    model: BrainMRIModule = BrainMRIModule.load_from_checkpoint(cfg.checkpoint)
    model.eval()
    if cfg.module.qat:
        model = quantization.convert(model)

    dataset = BrainMRIInferenceDataset(Path(cfg.dataset_dirpath), grayscale, cfg.image_size)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    with torch.no_grad():
        i = 0
        for input_images, images, image_paths in tqdm(dataloader):
            preds = model(input_images.to(model.device)).permute(0, 2, 3, 1)  # [N, H, W, C]
            preds = torch.where(preds > cfg.module.config.bin_det_threshold, 255.0, 0.0)

            for pred, image, image_path in zip(
                preds.cpu().numpy().astype(np.uint8), images.cpu().numpy().astype(np.uint8), image_paths
            ):
                filepath = str(output_path / f"{Path(image_path).stem}_{i}.jpeg")
                contours, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if grayscale else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
                cv2.imwrite(filepath, image)

                i += 1


if __name__ == "__main__":
    main()
