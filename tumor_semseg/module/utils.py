"""
Utilities to work with Lightning modules.
"""

# Third-Party
import lightning as L
import torch.nn as nn


class ExportableModel(nn.Module):
    """
    Auxiliary class to export models to ONNX due to Lightning module issues to export directly to ONNX.
    """

    def __init__(self, module: L.LightningModule):
        super().__init__()

        assert hasattr(module, "model"), "The inference must be contained within a standalone attribute `model`"

        self.model = module.model
        if hasattr(module, "qconfig"):
            self.quant = module.quant
            self.dequant = module.dequant

    def forward(self, inputs):
        if hasattr(self, "quant"):
            return self.dequant(self.model(self.quant(inputs)))
        return self.model(inputs)
