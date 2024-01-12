# Third-party
# Third-Party
import pytest
import torch
import torch.nn as nn
from torch.ao.nn.intrinsic.modules.fused import ConvReLU2d

# TumorSemSeg
from tumor_semseg.architecture.utils import auto_fuse_modules


@pytest.fixture
def fusable_model():
    return nn.Sequential(nn.Conv2d(1, 20, 5), nn.BatchNorm2d(20), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU()).eval()


@pytest.fixture
def not_fusable_model():
    return nn.Sequential(nn.Conv2d(1, 20, 5), nn.Sigmoid(), nn.Conv2d(20, 64, 5), nn.Tanh()).eval()


def test_fusion_in_model_with_fusable_layers(fusable_model):
    auto_fuse_modules(fusable_model)
    fused_layers = list(fusable_model.children())

    # Check the structure of the fused model
    expected_types = [ConvReLU2d, nn.Identity, nn.Identity, ConvReLU2d, nn.Identity]
    actual_types = [type(layer) for layer in fused_layers]

    assert actual_types == expected_types, "Fused model structure is not as expected"


def test_fusion_in_model_without_fusable_layers(not_fusable_model):
    original_layers = list(not_fusable_model.children())
    auto_fuse_modules(not_fusable_model)
    fused_layers = list(not_fusable_model.children())

    # Check if the number of layers remains the same
    assert len(fused_layers) == len(original_layers), "No fusion should occur in a model without fusable layers"


def test_fusion_does_not_alter_model_functionality(fusable_model):
    input_tensor = torch.randn(1, 1, 28, 28)
    original_output = fusable_model(input_tensor)

    auto_fuse_modules(fusable_model)
    fused_output = fusable_model(input_tensor)

    # Check if the output of the model remains the same after fusion
    torch.testing.assert_allclose(original_output, fused_output, atol=1e-6, rtol=1e-4)


def test_fusion_with_empty_model():
    model = nn.Sequential()
    model.eval()
    auto_fuse_modules(model)
    fused_layers = list(model.children())

    # Check if the model remains empty after fusion
    assert len(fused_layers) == 0, "Fusion on an empty model should not add layers"


def test_model_not_in_eval_mode(fusable_model):
    fusable_model.train()
    with pytest.raises(AssertionError):
        auto_fuse_modules(fusable_model)
