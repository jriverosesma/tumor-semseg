"""
This file defines the data module for the Brain MRI data.
"""

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import lightning as L
import numpy as np
import pandas as pd
from PIL import Image
from PIL.Image import Resampling
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

# Tumor SemSeg
from tumor_semseg.dataset.augmentations import get_training_augmentations


@dataclass
class BrainMRIDataModuleConfig:
    dataset_dirpath: Path
    seed: Optional[int] = None
    test_size: float = 0.2
    image_size: tuple[int, int] = field(default_factory=lambda: (256, 256))  # HxW
    batch_size: int = 8
    num_workers: int = 2
    augment: bool = True
    drop_last: bool = False


class BrainMRIDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        images: list[str],
        masks: list[str],
        augment: bool = True,
        image_size: tuple[int, int] = (256, 256),
    ):
        self.dataset = dataset
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.augment = augment
        self.augmentation_pipeline = get_training_augmentations(image_size)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB").resize(self.image_size, resample=Resampling.BILINEAR)
        mask = Image.open(self.masks[idx]).convert("L").resize(self.image_size, resample=Resampling.NEAREST)

        if self.augment and self.dataset == "train":
            augmented = self.augmentation_pipeline(image=np.array(image), mask=np.array(mask))
            image, mask = augmented["image"], augmented["mask"]

        return self.transform(image), self.transform(mask)


class BrainMRIDataModule(L.LightningDataModule):
    def __init__(self, config: BrainMRIDataModuleConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: str):
        if stage == "fit":
            assert (self.config.dataset_dirpath / "kaggle_3m").exists()

            mask_filepaths = glob.glob(str(self.config.dataset_dirpath) + "/kaggle_3m/**/*_mask.tif", recursive=True)
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

            self.train = BrainMRIDataset("train", x_train, y_train, self.config.augment, self.config.image_size)
            self.val = BrainMRIDataset("val", x_val, y_val, self.config.augment, self.config.image_size)

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
