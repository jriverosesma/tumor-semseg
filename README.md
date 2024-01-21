# Tumor Semantic Segmentation

[![Unit tests](https://github.com/jriverosesma/python-project/actions/workflows/unit_tests.yaml/badge.svg)](https://github.com/jriverosesma/python-project/actions/workflows/unit_tests.yaml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Lightning](https://img.shields.io/badge/Lightning-8A2BE2)](https://github.com/Lightning-AI/pytorch-lightning)
[![Hydra](https://img.shields.io/badge/Hydra-blue)](https://github.com/facebookresearch/hydra)
[![Aim](https://img.shields.io/badge/Aim-gray)](https://github.com/aimhubio/aim)


<div align="center">
    <img src="assets/pl.png" alt="PyTorch Lightning" width="200"> 
    <img src="assets/hydra.jpeg" alt="Hydra" width="200"> 
    <img src="assets/aimstack.png" alt="AimStack" width="200">
</div>

## Table of contents
1. [Overview](README.md#1-overview)  
2. [Installation](README.md#2-installation)  
3. [Quickstart](README.md#3-quickstart)
4. [Features](README.md#4-features)
5. [Running tests](README.md#5-running-tests)

## 1. Overview

This is a lightweight and flexible Semantic Segmentation framework for MRI tumor detection on [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

We can easily extend the framework to new datasets thanks to Lightning datamodules.

<br>
<div align="center">
    <table>
        <tr>
            <td><img src="assets/pred_1.jpeg" alt="Pred 1" width="200"/></td>
            <td><img src="assets/pred_2.jpeg" alt="Pred 2" width="200"/></td>
            <td rowspan="2"><img src="assets/gt_pred.png" alt="GT - Pred" width="500"></td>
        </tr>
        <tr>
            <td><img src="assets/pred_3.jpeg" alt="Pred 3" width="200"/></td>
            <td><img src="assets/pred_4.jpeg" alt="Pred 4" width="200"/></td>
        </tr>
    </table>
</div>

## 2. Installation

```bash
git clone https://github.com/jriverosesma/tumor-semantic-segmentation
conda create -n tumor-semseg python=3.10 --no-default-packages -y
conda activate tumor-semseg
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e .[all]
pre-commit install
```

## 3. Quickstart

### 3.1 Notebook

Check the Kaggle notebook [here](notebooks/kaggle_tutorial.ipynb) for a practical example on how to use the framework ⭐

### 3.2 Step by step

1. Follow the [installation instructions](README.md#2-installation).
2. Download and extract dataset from [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).
3. Specify the `dataset_dirpath` in the [main configuration file](tumor_semseg/configuration/main.yaml).
4. Train using `semseg_train` command. Run `aim up` in a different terminal to keep track of the training using Aim logger.
5. Replace `checkpoint` in the [main configuration file](tumor_semseg/configuration/main.yaml) by the path of the checkpoint saved after training. The best and last epoch models are saved by default in `.aim/<experiment_name>/<run_id>`.
6. Run model evaluation using `semseg_eval`.
7. Run inferences using `semseg_infer`.
8. Export the model to ONNX using `semseg_export`.

### 3.3 Pre-trained models

Replace `checkpoint` in the [main configuration file](tumor_semseg/configuration/main.yaml) by the path to the torch .ckpt checkpoint.

UNet++ with MobileNetv2 encoder:

- Torch checkpoint: [epoch=26-val_mIoU=0.91.ckpt](https://huggingface.co/jriverosesma/unetplusplus_mobilenetv2-encoder_epoch-26-val_mIoU-0.91/resolve/main/epoch%3D26-val_mIoU%3D0.91.ckpt?download=true)

- ONNX checkpoint: [epoch=26-val_mIoU=0.91.onnx](https://huggingface.co/jriverosesma/unetplusplus_mobilenetv2-encoder_epoch-26-val_mIoU-0.91/resolve/main/epoch%3D26-val_mIoU%3D0.91.onnx?download=true)

## 4. Features

- Training powered by [Lightning](https://github.com/Lightning-AI/pytorch-lightning) ⚡
- Easy configuration management using [Hydra](https://github.com/facebookresearch/hydra) ⚙️
- Supercharged logging using [Aim](https://github.com/aimhubio/aim) 🗃
- Ready to use scripts: 
    - Train from scratch or from a checkpoint: `semseg_train`.
    - Compute metrics for model evaluation: `semseg_eval`.
    - Run inference on images: `semseg_infer`.
    - Export model to ONNX: `semseg_export`.
- Direct integration with [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) via Hydra config: more than 124 encoders and many model architectures ready to use.
- Hyper-parameter tuning using Optuna plugin for Hydra.
- Model compression:
    - Quantized Aware Training using PyTorch quantization modules.
    - Pruning Aware Training using Lightning callbacks.

## 5. Running tests

Follow the [installation instructions](README.md#2-installation). Then run the following commands from the root of the repository.

```bash
pytest
```

**NOTE**: Make sure to active the conda environment with `conda activate tumor_semseg` before running the tests.
