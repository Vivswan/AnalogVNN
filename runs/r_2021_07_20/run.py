import torchvision
from torch import optim

from dataloaders.loss_functions import nll_loss_fn
from runs.r_2021_07_20.a_single_linear_layer_model import SingleLinearLayerModel
from runs.r_2021_07_20.b_reduce_precision_layer_model import ReducePrecisionLayerModel
from runs.r_2021_07_20.c_gaussian_reduce_precision_layer_model import GaussianNoiseReducePrecisionLayerModel

DEFAULT_PARAMETERS = {
    "dataset": torchvision.datasets.MNIST,
    "batch_size": 1000,
    "epochs": 10,
    "optimizer": optim.Adam,
    "loss_fn": nll_loss_fn,
    "model_kargs": {},
}

RUN_MODELS_20210720 = {
    "SingleLinear": {
        **DEFAULT_PARAMETERS,
        "model": SingleLinearLayerModel
    },
    "ReducePrecision-2": {
        **DEFAULT_PARAMETERS,
        "model": ReducePrecisionLayerModel,
        "model_kargs": dict(precision=2)
    },
    "ReducePrecision-4": {
        **DEFAULT_PARAMETERS,
        "model": ReducePrecisionLayerModel,
        "model_kargs": dict(precision=4)
    },
    "ReducePrecision-8": {
        **DEFAULT_PARAMETERS,
        "model": ReducePrecisionLayerModel,
        "model_kargs": dict(precision=8)
    },
    "ReducePrecision-2_GaussianNoise-0.1": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=2, std=0.1)
    },
    "ReducePrecision-2_GaussianNoise-0.05": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=2, std=0.05)
    },
    "ReducePrecision-4_GaussianNoise-0.1": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=4, std=0.1)
    },
    "ReducePrecision-4_GaussianNoise-0.05": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=4, std=0.05)
    },
    "ReducePrecision-8_GaussianNoise-0.1": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=8, std=0.1)
    },
    "ReducePrecision-8_GaussianNoise-0.05": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=8, std=0.05)
    },
    "ReducePrecision-16_GaussianNoise-0.1": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=16, std=0.1)
    },
    "ReducePrecision-16_GaussianNoise-0.05": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=16, std=0.05)
    },
}
