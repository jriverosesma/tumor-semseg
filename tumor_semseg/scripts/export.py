"""
This files defines the entry point for exporting models to ONNX format.
"""

# Standard
from pathlib import Path

# Third-Party
import hydra
import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
from omegaconf import DictConfig
from torch import _C, Tensor

# TumorSemSeg
from tumor_semseg.module.brain_mri_module import BrainMRIModule
from tumor_semseg.module.utils import ExportableModel


def check_onnx_model(onnx_model_path: str, expected_output: Tensor, input_tensor: Tensor) -> None:
    # Model check
    onnx.checker.check_model(onnx_model_path)

    # Inference check
    ort_session = ort.InferenceSession(onnx_model_path)

    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(ort_outputs[0], expected_output.numpy(), rtol=1e-03, atol=1e-05)


@hydra.main(config_path="../configuration", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    module = BrainMRIModule.load_from_checkpoint(cfg.checkpoint)
    if hasattr(module, "qconfig"):
        module = module.get_quantized_model()
    exportable_model: ExportableModel = ExportableModel(module)
    exportable_model.eval()

    with torch.no_grad():
        model_output = exportable_model(module.example_input_array)

    save_filepath = (
        str(Path(cfg.checkpoint).parent / (Path(cfg.checkpoint).stem + ".onnx"))
        if cfg.export.save_filepath is None
        else cfg.export.save_filepath
    )

    # Export from torch to ONNX
    torch.onnx.export(
        exportable_model,
        module.example_input_array,
        save_filepath,
        export_params=True,
        opset_version=cfg.export.opset_version,
        training=_C._onnx.TrainingMode.EVAL,
        do_constant_folding=cfg.export.do_constant_folding,
        input_names=cfg.export.input_names,
        output_names=cfg.export.output_names,
    )

    check_onnx_model(save_filepath, model_output, module.example_input_array)

    # Simplify ONNX model
    simplified_model, check = onnxsim.simplify(
        save_filepath,
        check_n=cfg.export.check_n,
        perform_optimization=cfg.export.perform_optimization,
        skip_fuse_bn=cfg.export.skip_fuse_bn,
        skip_constant_folding=not cfg.export.do_constant_folding,
    )

    assert check, "Simplified ONNX model could not be validated"

    onnx.save(simplified_model, save_filepath)
    check_onnx_model(save_filepath, model_output, module.example_input_array)


if __name__ == "__main__":
    main()
