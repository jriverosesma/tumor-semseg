import pytest
import torch
from torch import nn

# Tumor SemSeg
from tumor_semseg.architecture.deeplab import DeepLabv3, DeepLabv3Config


@pytest.mark.parametrize(
    "input_shape, config",
    [
        ((2, 3, 256, 256), DeepLabv3Config(10, 3, "mobilenet", True)),
        ((2, 3, 256, 256), DeepLabv3Config(5, 3, "mobilenet", False)),
        ((2, 3, 256, 256), DeepLabv3Config(10, 3, "resnet50", False)),
        ((2, 3, 256, 256), DeepLabv3Config(5, 3, "resnet50", True)),
        ((2, 3, 256, 256), DeepLabv3Config(10, 3, "resnet101", False)),
        ((2, 3, 256, 256), DeepLabv3Config(5, 3, "resnet101", True)),
    ],
)
def test_deeplab(input_shape, config):
    net = DeepLabv3(config)
    output = net(torch.rand(input_shape))

    expected_output_shape = (input_shape[0], config.n_classes, input_shape[2], input_shape[3])

    assert output.shape == expected_output_shape
