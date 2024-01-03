# Tumor Semantic Segmentation

[![Unit tests](https://github.com/jriverosesma/python-project/actions/workflows/unit_tests.yaml/badge.svg)](https://github.com/jriverosesma/python-project/actions/workflows/unit_tests.yaml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Lightning](https://img.shields.io/badge/Lightning-8A2BE2)](https://github.com/Lightning-AI/pytorch-lightning)
[![Hydra](https://img.shields.io/badge/Hydra-blue)](https://github.com/facebookresearch/hydra)
[![Aim](https://img.shields.io/badge/Aim-gray)](https://github.com/aimhubio/aim)


<div align="center">
    <img src="assets/pl.png" width="200"> <img src="assets/hydra.jpeg" width="200"> <img src="assets/aimstack.png" width="200">
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

1. Follow the [installation instructions](README.md#2-installation).
2. Download and extract dataset from [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).
3. Specify the `dataset_dirpath` in the [main configuration file](tumor_semseg/configuration/main.yaml).
4. Train using `semseg_train` command. Run `aim up` in a different terminal to keep track of the training using Aim logger.
5. Replace `checkpoint` in the [main configuration file](tumor_semseg/configuration/main.yaml) by the path of the checkpoint saved after training. The best and last epoch models are saved by default in `.aim/<experiment_name>/<run_id>`.
6. Run model evaluation using `semseg_eval`.
7. Run inferences using `semseg_infer`.
8. Export the model to ONNX using `semseg_export`.

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
