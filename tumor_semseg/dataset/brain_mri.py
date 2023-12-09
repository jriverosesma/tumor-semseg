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
from torchvision import transforms


@dataclass
class BrainMRIDataModuleConfig:
    dataset_dir: Path
    seed: Optional[int] = None
    test_size: float = 0.2
    image_size: list[int] = field(default_factory=lambda: [256, 256])  # HxW
    batch_size: int = 8
    num_workers: int = 2
    drop_last: bool = False


class BrainMRIDataset(Dataset):
    def __init__(self, images: list[str], masks: list[str], image_size=tuple[256, 256]):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB").resize(self.image_size, resample=Resampling.BILINEAR)
        mask = Image.open(self.masks[idx]).resize(self.image_size, resample=Resampling.NEAREST)
        image = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.float32) / 255

        return self.transform(image), self.transform(mask)


class BrainMRIDataModule(L.LightningDataModule):
    def __init__(self, config: BrainMRIDataModuleConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: str):
        if stage == "fit":
            self.train = BrainMRIDataset(self.x_train, self.y_train, self.config.image_size)
            self.val = BrainMRIDataset(self.x_val, self.y_val, self.config.image_size)

    def prepare_data(self) -> None:
        mask_filepaths = glob.glob(str(self.config.dataset_dir) + "/**/*_mask.tif", recursive=True)
        image_filepaths = [path.replace("_mask", "") for path in mask_filepaths]

        df = pd.DataFrame(data={"image_filepaths": image_filepaths, "mask_filepaths": mask_filepaths})
        df = df.sort_values("image_filepaths")
        df["patient"] = df["mask_filepaths"].apply(lambda x: Path(x).parent.name)
        df["positive"] = df["mask_filepaths"].apply(lambda x: np.any(cv2.imread(x, 0)))

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            df["image_filepaths"].values,
            df["mask_filepaths"].values,
            test_size=self.config.test_size,
            random_state=self.config.seed,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=self.config.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=self.config.drop_last,
        )
