"""
DeepLabv3 model architercture definition.
"""

# Standard
from dataclasses import dataclass
from enum import Enum

# Third-Party
import torch.nn as nn
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
)


class DeepLabv3Version(Enum):
    mobilenet = (deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    resnet50 = (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights.DEFAULT)
    resnet101 = (deeplabv3_resnet101, DeepLabV3_ResNet101_Weights.DEFAULT)


@dataclass
class DeepLabv3Config:
    n_classes: int
    in_channels: int = 3
    model_version_key: str = "mobilenet"
    pretrained: bool = False


class DeepLabv3(nn.Module):
    def __init__(self, config: DeepLabv3Config):
        super().__init__()
        model_version = DeepLabv3Version[config.model_version_key]
        self.model = model_version.value[0](weights=model_version.value[1] if config.pretrained else None)
        if config.pretrained:
            self.model.aux_classifier = None
        self.model.classifier[-1] = nn.Conv2d(256, config.n_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)["out"]
