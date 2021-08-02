import copy

import torchvision
from torch import optim

from dataloaders.loss_functions import nll_loss_fn
from runs.r_2021_07_31.double_linear_layer_model import DoubleLinearLayerModel
from runs.r_2021_07_31.single_linear_layer_model import SingleLinearLayerModel
from runs.r_2021_07_31.triple_linear_layer_model import TripleLinearLayerModel

DEFAULT_PARAMETERS = {
    "dataset": torchvision.datasets.MNIST,
    "batch_size": 1000,
    "epochs": 10,
    "optimizer": optim.Adam,
    "loss_fn": nll_loss_fn,
    "model_kargs": {},
}

RUN_MODELS = {
    "SingleLinear": {
        **DEFAULT_PARAMETERS,
        "model": SingleLinearLayerModel,
    },
    "DoubleLinear": {
        **DEFAULT_PARAMETERS,
        "model": DoubleLinearLayerModel
    },
    "TripleLinear": {
        **DEFAULT_PARAMETERS,
        "model": TripleLinearLayerModel
    },
}

temp_models = {}

for i in RUN_MODELS:
    for j in RUN_MODELS[i]["model"].approaches:
        temp_models[f"{i}-{j.value}"] = copy.deepcopy(RUN_MODELS[i])
        temp_models[f"{i}-{j.value}"]["model_kargs"]["approach"] = j

RUN_MODELS = temp_models
temp_models = {}

for i in RUN_MODELS:
    if i[-2:] != "FA":
        temp_models[i] = {**RUN_MODELS[i]}
        continue

    for j in [1, 0.1, 0.01]:
        temp_models[f"{i}-{j:1.2f}"] = copy.deepcopy(RUN_MODELS[i])
        temp_models[f"{i}-{j:1.2f}"]["model_kargs"]["std"] = j

RUN_MODELS = temp_models
del temp_models
