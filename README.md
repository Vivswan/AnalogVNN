# AnalogVNN

[![arXiv](https://img.shields.io/badge/arXiv-2210.10048-orange.svg)](https://arxiv.org/abs/2210.10048)
[![AML](https://img.shields.io/badge/AML-10.1063/5.0134156-orange.svg)](https://doi.org/10.1063/5.0134156)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vivswan/AnalogVNN/blob/release/docs/_static/AnalogVNN_Demo.ipynb)

[![PyPI version](https://badge.fury.io/py/analogvnn.svg)](https://badge.fury.io/py/analogvnn)
[![Documentation Status](https://readthedocs.org/projects/analogvnn/badge/?version=stable)](https://analogvnn.readthedocs.io/en/stable/?badge=stable)
[![Python](https://img.shields.io/badge/python-3.7--3.11-blue)](https://badge.fury.io/py/analogvnn)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-blue.svg)](https://opensource.org/licenses/MPL-2.0)

Documentation: [https://analogvnn.readthedocs.io/](https://analogvnn.readthedocs.io/)

## Installation:

- Install [PyTorch](https://pytorch.org/)
- Install AnalogVNN using [pip](https://pypi.org/project/analogvnn/)

```bash
  # Current stable release for CPU and GPU
  pip install analogvnn
  
  # For additional optional features
  pip install analogvnn[full]
```

## Usage:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vivswan/AnalogVNN/blob/release/docs/_static/AnalogVNN_Demo.ipynb)

- Sample code with AnalogVNN: [sample_code.py](https://github.com/Vivswan/AnalogVNN/blob/release/sample_code.py)
- Sample code without
  AnalogVNN: [sample_code_non_analog.py](https://github.com/Vivswan/AnalogVNN/blob/release/sample_code_non_analog.py)
- Sample code with AnalogVNN and
  Logs: [sample_code_with_logs.py](https://github.com/Vivswan/AnalogVNN/blob/release/sample_code_with_logs.py)
- Jupyter
  Notebook: [AnalogVNN_Demo.ipynb](https://github.com/Vivswan/AnalogVNN/blob/release/docs/_static/AnalogVNN_Demo.ipynb)

## Abstract

![3 Layered Linear Photonic Analog Neural Network](docs/_static/analogvnn_model.png)

[//]: # (![3 Layered Linear Photonic Analog Neural Network]&#40;https://github.com/Vivswan/AnalogVNN/raw/release/docs/_static/analogvnn_model.png&#41;)

**AnalogVNN** is a simulation framework built on PyTorch which can simulate the effects of
optoelectronic noise, limited precision, and signal normalization present in photonic
neural network accelerators. We use this framework to train and optimize linear and
convolutional neural networks with up to 9 layers and ~1.7 million parameters, while
gaining insights into how normalization, activation function, reduced precision, and
noise influence accuracy in analog photonic neural networks. By following the same layer
structure design present in PyTorch, the AnalogVNN framework allows users to convert most
digital neural network models to their analog counterparts with just a few lines of code,
taking full advantage of the open-source optimization, deep learning, and GPU acceleration
libraries available through PyTorch.

AnalogVNN Paper: [https://doi.org/10.1063/5.0134156](https://doi.org/10.1063/5.0134156)

## Citing AnalogVNN

We would appreciate if you cite the following paper in your publications for which you used AnalogVNN:

```bibtex
@article{shah2023analogvnn,
  title={AnalogVNN: A fully modular framework for modeling and optimizing photonic neural networks},
  author={Shah, Vivswan and Youngblood, Nathan},
  journal={APL Machine Learning},
  volume={1},
  number={2},
  year={2023},
  publisher={AIP Publishing}
}
```

Or in textual form:

```text
Vivswan Shah, and Nathan Youngblood. "AnalogVNN: A fully modular framework for modeling 
and optimizing photonic neural networks." APL Machine Learning 1.2 (2023).
DOI: 10.1063/5.0134156
```