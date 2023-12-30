"""
Utilities the work with Neural Network architectures.
"""

# Third-Party
import torch.ao.quantization as quantization

# Third-party
import torch.nn as nn

# Fusable patterns sorted by priority
FUSABLE_PATTERNS = [
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU),
    (nn.Conv2d, nn.ReLU),
    (nn.Linear, nn.BatchNorm2d),
    (nn.Linear, nn.ReLU),
]


def auto_fuse_modules(model: nn.Module) -> None:
    """
    Automatically fuses sets of `nn.Module` based on the available fusable patterns.

    NOTE: This is an experimental feature.

    Args:
        model: Module to fuse.
    """

    named_modules: list[tuple[str, nn.Module]] = list(model.named_modules())
    modules_to_fuse: list[list[str]] = []
    modules_to_fuse_idx: list[int] = []
    for i in range(len(named_modules)):
        if i in modules_to_fuse_idx:
            continue
        for pattern in FUSABLE_PATTERNS:
            try:
                parent: str = ""
                matching_modules: list[str] = []
                matching_modules_idx: list[int] = []
                for j, pattern_module in enumerate(pattern):
                    if isinstance(named_modules[i + j][1], pattern_module):
                        current_module_parent = named_modules[i + j][0].rsplit(".", 1)[0]
                        if not parent:
                            parent = current_module_parent
                        elif parent != current_module_parent:
                            break
                        matching_modules.append(named_modules[i + j][0])
                        matching_modules_idx.append(i + j)
                    else:
                        break
                if matching_modules and len(matching_modules) > 1:
                    modules_to_fuse.append(matching_modules)
                    modules_to_fuse_idx += matching_modules_idx
                    break
            except IndexError:
                continue

    quantization.fuse_modules(model, modules_to_fuse, inplace=True)
