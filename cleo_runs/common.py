import copy
import inspect

import torch
from torch import optim

from nn.utils.make_dot import make_dot

TENSORBOARD = True
cleo_default_parameters = {
    'optimiser_class': optim.Adam,
    'optimiser_parameters': {},
    'batch_size': 128,
    'epochs': 10,
}


def save_graph(filename, output, named_parameters):
    make_dot(output, params=dict(named_parameters)).render(filename, cleanup=True)


def cross_entropy_loss_accuracy(output, target):
    _, preds = torch.max(output.data, 1)
    correct = (preds == target).sum().item()
    return correct / len(output)


def combination_parameters(para: dict, name: str, list_para: list, contain: str = None, sufix: str = None):
    temp = {}
    for key, value in para.items():
        if contain is not None and contain not in key:
            temp[key] = value
            continue
        if sufix is not None and not key.endswith(sufix):
            temp[key] = value
            continue

        for v in list_para:
            temp_value = copy.deepcopy(value)
            temp_value[name] = v
            if inspect.isclass(v):
                temp[f"{key}_{v.__name__}"] = temp_value
            else:
                temp[f"{key}_{str(v)}"] = temp_value

    return temp


def combination_parameters_sequence(start_dict, *args):
    parameter = copy.deepcopy(start_dict)
    for i in args:
        parameter = combination_parameters(parameter, *i)
    return parameter


def run_function_with_parameters(func, parameters, continue_from=None):
    c = False
    for i, (name, p) in enumerate(parameters.items()):
        if continue_from is not None and len(continue_from) > 0:
            if name == continue_from:
                c = True
            if not c:
                continue

        print()
        print(f"{i + 1}/{len(parameters.keys())}")
        func(name, **p)
