import pytest
import torch
from torch import nn

# Tumor SemSeg
from tumor_semseg.architecture.unet import UNet


@pytest.mark.parametrize(
    "input_shape, n_classes, activation",
    [
        ((5, 3, 256, 256), 10, nn.ReLU(inplace=True)),
        ((1, 3, 256, 256), 5, nn.ReLU(inplace=True)),
        ((10, 1, 256, 256), 1, nn.Sigmoid()),
        ((10, 1, 64, 64), 1, nn.Sigmoid()),
        ((10, 1, 256, 128), 3, nn.Sigmoid()),
    ],
)
def test_unet(input_shape, n_classes, activation):
    net = UNet(n_classes, input_shape[1], activation)
    output = net(torch.rand(input_shape))

    expected_output_shape = (input_shape[0], n_classes, input_shape[2], input_shape[3])

    assert output.shape == expected_output_shape
