"""
Utilities to work with Lightning modules.
"""

# Third-Party
import torch.nn as nn

# TumorSemSeg
from tumor_semseg.module.brain_mri_module import BrainMRIModule


class ExportableModel(nn.Module):
    """
    Auxiliary class to export models to ONNX due to Lightning module issues to export directly to ONNX.
    """

    def __init__(self, module: BrainMRIModule):
        super().__init__()
        self.model = module.model
        if hasattr(module, "qconfig"):
            self.quant = module.quant
            self.dequant = module.dequant

    def forward(self, inputs):
        if hasattr(self, "quant"):
            return self.dequant(self.model(self.quant(inputs)))
        return self.model(inputs)
