"""
This file defines the data module for the Brain MRI data.
"""

# Standard
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Third-Party
import albumentations as A
import cv2
import lightning as L
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from PIL.Image import Resampling
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# TumorSemSeg
from tumor_semseg.data.augmentations import get_weak_augmentations


@dataclass
class BrainMRIDataModuleConfig:
    dataset_dirpath: Path
    seed: Optional[int] = None
    test_size: float = 0.1
    image_size: tuple[int, int] = field(default_factory=lambda: (256, 256))  # HxW
    in_channels: int = 3
    batch_size: int = 32
    num_workers: int = 1
    augment: bool = True
    drop_last: bool = False

    def __post_init__(self):
        # NOTE: OmegaConfg does not currently support `tuple`
        self.image_size = tuple(self.image_size)


class BrainMRIDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        images: list[str],
        masks: list[str],
        augment: bool,
        grayscale: bool,
        image_size: tuple[int, int],
    ):
        self.dataset = dataset
        self.images = images
        self.masks = masks
        self.grayscale = grayscale
        self.image_size = image_size
        self.transform = A.Compose(
            [A.Normalize(p=1.0, mean=0.5, std=0.5) if self.grayscale else A.Normalize(p=1.0), ToTensorV2()]
        )
        if augment:
            self.transform = A.Compose([get_weak_augmentations(), self.transform])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = (
            Image.open(self.images[idx])
            .convert("L" if self.grayscale else "RGB")
            .resize(self.image_size, resample=Resampling.BILINEAR)
        )
        mask = Image.open(self.masks[idx]).convert("L").resize(self.image_size, resample=Resampling.NEAREST)

        image = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.float32) / 255.0

        transformed = self.transform(image=image, mask=mask)

        return transformed["image"], transformed["mask"]


class BrainMRIDataModule(L.LightningDataModule):
    def __init__(self, config: BrainMRIDataModuleConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: str):
        if stage == "fit":
            assert (self.config.dataset_dirpath).exists()

            mask_filepaths = glob.glob(str(self.config.dataset_dirpath) + "/**/*_mask.tif", recursive=True)
            image_filepaths = [path.replace("_mask", "") for path in mask_filepaths]

            df = pd.DataFrame(data={"image_filepaths": image_filepaths, "mask_filepaths": mask_filepaths})
            df = df.sort_values("image_filepaths")
            df["patient"] = df["mask_filepaths"].apply(lambda x: Path(x).parent.name)
            df["positive"] = df["mask_filepaths"].apply(lambda x: np.any(cv2.imread(x, 0)))

            x_train, x_val, y_train, y_val = train_test_split(
                df["image_filepaths"].values,
                df["mask_filepaths"].values,
                test_size=self.config.test_size,
                random_state=self.config.seed,
                stratify=df["positive"].values,
            )

            match self.config.in_channels:
                case 3:
                    grayscale = False
                case 1:
                    grayscale = True
                case _:
                    raise KeyError("in_channels must be one 3 (RGB) or 1 ', 'sum', 'none']")

            self.train = BrainMRIDataset(
                "train",
                x_train,
                y_train,
                augment=self.config.augment,
                grayscale=grayscale,
                image_size=self.config.image_size,
            )
            self.val = BrainMRIDataset(
                "val", x_val, y_val, augment=False, grayscale=grayscale, image_size=self.config.image_size
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=self.config.drop_last,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=self.config.drop_last,
            persistent_workers=True,
        )
