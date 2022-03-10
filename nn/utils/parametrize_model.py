import torch
from torch import nn
from torch.nn.utils import parametrize


def parametrize_model(model: nn.Module, transform):
    with torch.no_grad():
        if len(list(model.parameters())) != len(list(model.named_parameters())):
            raise Exception("All parameters need to be named to be converted.")

        for name, parameter in list(model.named_parameters()):
            if not parameter.requires_grad:
                continue

            parametrize.register_parametrization(model, name, transform)
