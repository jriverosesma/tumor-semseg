# Third-Party
import pytest
import torch
from torch import nn

# TumorSemSeg
from tumor_semseg.architecture.unet import UNet, UNetConfig


@pytest.mark.parametrize(
    "input_shape, config",
    [
        ((5, 3, 256, 256), UNetConfig(10, 3, "ReLU", {"inplace": True}, True)),
        ((1, 3, 256, 256), UNetConfig(5, 3, "ReLU", {"inplace": True}, False)),
        ((10, 1, 256, 256), UNetConfig(7, 1, "Sigmoid", True)),
        ((10, 1, 64, 64), UNetConfig(4, 1, "Sigmoid", False)),
        ((10, 1, 256, 128), UNetConfig(1, 1, "Sigmoid", True)),
    ],
)
def test_unet(input_shape, config):
    net = UNet(config)
    output = net(torch.rand(input_shape))

    expected_output_shape = (input_shape[0], config.n_classes, input_shape[2], input_shape[3])

    assert output.shape == expected_output_shape
