"""
This files defines the entry point for training.
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


def check_onnx_model(onnx_model_path: Path, expected_output: Tensor, input_tensor: Tensor) -> None:
    # Model check
    onnx.checker.check_model(onnx_model_path)

    # Inference check
    ort_session = ort.InferenceSession(onnx_model_path)

    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(ort_outputs[0], expected_output.numpy(), rtol=1e-03, atol=1e-05)


@hydra.main(config_path="../configuration", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    brain_mri_model: BrainMRIModule = BrainMRIModule.load_from_checkpoint(cfg.checkpoint)
    brain_mri_model.eval()

    # if cfg.export.filepath is None:
    # save_filepath = Path(cfg.checkpoint).parent / (Path(cfg.checkpoint).stem + ".onnx")
    save_filepath = Path(cfg.checkpoint).parent / (Path(cfg.checkpoint).stem + ".onnx")

    with torch.no_grad():
        torch_output = brain_mri_model(brain_mri_model.example_input_array)

    # Export from torch to ONNX
    torch.onnx.export(
        brain_mri_model,
        brain_mri_model.example_input_array,
        save_filepath,
        export_params=True,
        verbose=True,
        opset_version=13,
        training=_C._onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    check_onnx_model(save_filepath, torch_output, brain_mri_model.example_input_array)

    # Simplify ONNX model
    simplified_model, check = onnxsim.simplify(
        save_filepath,
        check_n=10,
        perform_optimization=True,
        skip_fuse_bn=False,
        skip_constant_folding=False,
    )

    assert check, "Simplified ONNX model could not be validated"

    onnx.save(simplified_model, save_filepath)
    check_onnx_model(save_filepath, torch_output, brain_mri_model.example_input_array)


if __name__ == "__main__":
    main()
