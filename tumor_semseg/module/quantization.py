"""
Utilities for Neural Network quantization.
"""

# Third-Party
import torch
import torch.nn as nn

FUSABLE_PATTERNS = [
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU),
    (nn.Conv2d, nn.ReLU),
    (nn.Linear, nn.BatchNorm2d),
    (nn.Linear, nn.ReLU),
]


def get_fusable_layers(model: nn.Module):
    layers_to_fuse = []
    i = 0
    while i < len(model) - 1:
        name, layer = model[i]
        next_name, next_layer = model[i + 1]

        for pattern in FUSABLE_PATTERNS:
            if isinstance(layer, pattern[0]) and isinstance(next_layer, pattern[1]):
                fuse_layers = [name, next_name]

                # Check for optional third layer in the pattern
                if len(pattern) == 3 and i + 2 < len(model) and isinstance(model[i + 2][1], pattern[2]):
                    fuse_layers.append(model[i + 2][0])
                    i += 1  # Include third layer in fusion

                layers_to_fuse.append(fuse_layers)
                i += 1  # Skip next layer as it is included in fusion
                break  # Exit loop once a pattern is matched

        i += 1

    return layers_to_fuse


def auto_fuse_modules(model: nn.Module):
    for _, child in model.named_children():
        child_layers_to_fuse = get_fusable_layers(list(child.named_children()))
        if child_layers_to_fuse:
            torch.ao.quantization.fuse_modules(child, child_layers_to_fuse)
        auto_fuse_modules(child)
