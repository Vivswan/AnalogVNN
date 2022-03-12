import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList


def parametrize_model(model: nn.Module, transform):
    if isinstance(model, ParametrizationList):
        return
    print(model.__class__.__name__)
    with torch.no_grad():
        if len(list(model.parameters())) != len(list(model.named_parameters())):
            raise Exception("All parameters need to be named to be converted.")

        for name, parameter in list(model.named_parameters(recurse=False)):
            if not parameter.requires_grad:
                continue
            if parametrize.is_parametrized(model, name):
                continue

            parametrize.register_parametrization(model, name, transform)

        for module in model.children():
            if module == model:
                continue
            parametrize_model(module, transform=transform)
