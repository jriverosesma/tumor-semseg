[![Unit tests](https://github.com/jriverosesma/python-project/actions/workflows/unit_tests.yaml/badge.svg)](https://github.com/jriverosesma/python-project/actions/workflows/unit_tests.yaml)

# Tumor Semantic Segmentation

## Table of contents
1. [Overview](README.md#1-overview)  
2. [Installation](README.md#2-installation)  
3. [Running tests](README.md#3-running-tests)

## 1. Overview

Implementation of Semantic Segmentation NN for tumor detection on <dataset-name> dataset.

## 2. Installation

```bash
git clone https://github.com/jriverosesma/tumor-semantic-segmentation
conda create -y -n tumor_semseg python=3.10 --no-default-packages
conda activate tumor_semseg
python -m pip install --upgrade pip
pip install -e .[all]
pre-commit install
```

## 3. Running tests

Follow the installation instructions. Then run the following commands from the root of the repository.

```bash
pytest
```

NOTE: Make sure to active the conda environment with `conda activate tumor_semseg` before running the tests.
