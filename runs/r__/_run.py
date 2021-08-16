import copy
import json

import torchvision
from torch import optim

from dataloaders.loss_functions import nll_loss_fn
from nn.activations.ReLU import ReLU, LeakyReLU
from nn.activations.Tanh import Tanh
from nn.layers.Normalize import Clamp, Norm
from runs.r__.double_linear_layer_model import DoubleLinearLayerModel
from runs.r__.single_linear_layer_model import SingleLinearLayerModel
from runs.r__.triple_linear_layer_model import TripleLinearLayerModel

DEFAULT_PARAMETERS = {
    "dataset": torchvision.datasets.MNIST,
    "batch_size": 128,
    "epochs": 10,
    "optimizer": optim.Adam,
    "loss_fn": nll_loss_fn,
    "normalizer_fn": None,
    "model_kargs": {},
}
RUN_MODELS = {
    "SingleLinear": {
        **DEFAULT_PARAMETERS,
        "model_class": SingleLinearLayerModel,
    },
    "DoubleLinear": {
        **DEFAULT_PARAMETERS,
        "model_class": DoubleLinearLayerModel
    },
    "TripleLinear": {
        **DEFAULT_PARAMETERS,
        "model_class": TripleLinearLayerModel
    },
}

temp_models = {}
for i in RUN_MODELS:
    for j in [
        torchvision.datasets.MNIST,
        torchvision.datasets.FashionMNIST,
        # torchvision.datasets.CIFAR10,
        # torchvision.datasets.CIFAR100,
    ]:
        temp_models[f"{i}-{j.__name__}"] = copy.deepcopy(RUN_MODELS[i])
        temp_models[f"{i}-{j.__name__}"]["dataset"] = j
RUN_MODELS = temp_models

temp_models = {}
for i in RUN_MODELS:
    for j in RUN_MODELS[i]["model_class"].approaches:
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

temp_models = {}
for i in RUN_MODELS:
    for j in [
        ReLU,
        LeakyReLU,
        Tanh,
    ]:
        suffix = f"-{j.__name__}-"
        temp_models[f"{i}{suffix}"] = copy.deepcopy(RUN_MODELS[i])
        temp_models[f"{i}{suffix}"]["model_kargs"]["activation_class"] = j
RUN_MODELS = temp_models

temp_models = {}
for i in RUN_MODELS:
    for nfn in [Clamp, Norm]:
        temp_models[f"{i}{nfn.__name__}"] = copy.deepcopy(RUN_MODELS[i])
        temp_models[f"{i}{nfn.__name__}"]["normalizer_fn"] = nfn
RUN_MODELS = temp_models

del temp_models

if __name__ == '__main__':
    print(json.dumps(list(RUN_MODELS.keys()), indent=4))
    print(len(list(RUN_MODELS.keys())))
