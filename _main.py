import copy
import inspect
import math
import os
import shutil
import time
from enum import Enum
from math import exp

import torch
from torch import nn, optim

from nn.BaseModel import BaseModel
from nn.activations.Activation import InitImplement
from nn.activations.ELU import ELU
from nn.activations.Gaussian import GeLU
from nn.activations.Identity import Identity
from nn.activations.ReLU import LeakyReLU, ReLU
from nn.activations.SiLU import SiLU
from nn.activations.Tanh import Tanh
from nn.layers.Linear import Linear
from nn.layers.Normalize import Norm, Clamp
from nn.layers.ReducePrecision import ReducePrecision
from nn.layers.StochasticReducePrecision import StochasticReducePrecision
from nn.optimizer.ReducePrecisionOptimizer import ReducePrecisionOptimizer
from nn.utils.is_using_cuda import get_device, set_device
from nn.utils.make_dot import make_dot


class BackPassTypes(Enum):
    default = "default"
    BP = "BP"
    FA = "FA"
    DFA = "DFA"
    RFA = "RFA"
    RDFA = "RDFA"


def save_graph(filename, output, named_parameters):
    make_dot(output, params=dict(named_parameters)).render(filename + ".svg", cleanup=True)


class TestModel(BaseModel):
    def __init__(
            self,
            approach: BackPassTypes,
            std,
            activation_class,
            output_normalize_class,
            output_reduce_precision,
            output_reduce_precision_p
    ):
        super(TestModel, self).__init__()
        self.approach = approach
        self.std = std
        self.activation_class = activation_class
        self.output_normalize_class = output_normalize_class
        self.output_reduce_precision = output_reduce_precision
        self.output_reduce_precision_p = output_reduce_precision_p

        self.linear1 = Linear(in_features=64, out_features=16)
        self.output_reduce_precision1 = output_reduce_precision() if output_reduce_precision_p is None else output_reduce_precision(
            precision=output_reduce_precision_p)
        self.normalize_layer1 = output_normalize_class()
        self.activation1 = activation_class()

        self.linear2 = Linear(in_features=16, out_features=8)
        self.output_reduce_precision2 = output_reduce_precision() if output_reduce_precision_p is None else output_reduce_precision(
            precision=output_reduce_precision_p)
        self.normalize_layer2 = output_normalize_class()
        self.activation2 = activation_class()

        self.linear3 = Linear(in_features=8, out_features=4)
        self.output_reduce_precision3 = output_reduce_precision() if output_reduce_precision_p is None else output_reduce_precision(
            precision=output_reduce_precision_p)
        self.normalize_layer3 = output_normalize_class()
        self.activation3 = activation_class()

        if issubclass(activation_class, InitImplement):
            activation_class.initialise_(None, self.linear1.weight)
            activation_class.initialise_(None, self.linear2.weight)
            activation_class.initialise_(None, self.linear3.weight)
        elif issubclass(activation_class, nn.ReLU):
            nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity="relu")
            nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity="relu")
            nn.init.kaiming_uniform_(self.linear3.weight, nonlinearity="relu")
        elif issubclass(activation_class, nn.LeakyReLU):
            nn.init.kaiming_uniform_(self.linear1.weight, a=math.sqrt(5), nonlinearity="leaky_relu")
            nn.init.kaiming_uniform_(self.linear2.weight, a=math.sqrt(5), nonlinearity="leaky_relu")
            nn.init.kaiming_uniform_(self.linear3.weight, a=math.sqrt(5), nonlinearity="leaky_relu")
        else:
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
            nn.init.xavier_uniform_(self.linear3.weight)

        if approach == BackPassTypes.default:
            self.backward.use_default_graph = True

        if approach == BackPassTypes.BP:
            self.backward.add_relation(
                self.backward.OUTPUT,

                self.activation3,
                self.normalize_layer3,
                self.output_reduce_precision3,
                self.linear3.backpropagation,

                self.activation2,
                self.normalize_layer2,
                self.output_reduce_precision2,
                self.linear2.backpropagation,

                self.activation1,
                self.normalize_layer1,
                self.output_reduce_precision1,
                self.linear1.backpropagation
            )

        if approach == BackPassTypes.FA or approach == BackPassTypes.RFA:
            self.backward.add_relation(
                self.backward.OUTPUT,

                self.activation3,
                self.normalize_layer3,
                self.output_reduce_precision3,
                self.linear3.feedforward_alignment,

                self.activation2,
                self.normalize_layer2,
                self.output_reduce_precision2,
                self.linear2.feedforward_alignment,

                self.activation1,
                self.normalize_layer1,
                self.output_reduce_precision1,
                self.linear1.feedforward_alignment
            )

        if approach == BackPassTypes.DFA or approach == BackPassTypes.RDFA:
            self.backward.add_relation(
                self.backward.OUTPUT,
                self.linear3.direct_feedforward_alignment.pre_backward,
                self.activation3,
                self.normalize_layer3,
                self.output_reduce_precision3,
                self.linear3.direct_feedforward_alignment
            )
            self.backward.add_relation(
                self.backward.OUTPUT,
                self.linear2.direct_feedforward_alignment.pre_backward,
                self.activation2,
                self.normalize_layer2,
                self.output_reduce_precision2,
                self.linear2.direct_feedforward_alignment
            )
            self.backward.add_relation(
                self.backward.OUTPUT,
                self.linear1.direct_feedforward_alignment.pre_backward,
                self.activation1,
                self.normalize_layer1,
                self.output_reduce_precision1,
                self.linear1.direct_feedforward_alignment
            )

        if approach == BackPassTypes.RFA or approach == BackPassTypes.RDFA:
            self.linear1.feedforward_alignment.is_fixed = False
            self.linear2.feedforward_alignment.is_fixed = False
            self.linear3.feedforward_alignment.is_fixed = False
            self.linear1.direct_feedforward_alignment.is_fixed = False
            self.linear2.direct_feedforward_alignment.is_fixed = False
            self.linear3.direct_feedforward_alignment.is_fixed = False

        if std is not None:
            self.linear1.feedforward_alignment.std = std
            self.linear2.feedforward_alignment.std = std
            self.linear3.feedforward_alignment.std = std
            self.linear1.direct_feedforward_alignment.std = std
            self.linear2.direct_feedforward_alignment.std = std
            self.linear3.direct_feedforward_alignment.std = std

    def forward(self, x):
        x = self.linear1(x)
        x = self.output_reduce_precision1(x)
        x = self.normalize_layer1(x)
        x = self.activation1(x)

        x = self.linear2(x)
        x = self.output_reduce_precision2(x)
        x = self.normalize_layer2(x)
        x = self.activation2(x)

        x = self.linear3(x)
        x = self.output_reduce_precision3(x)
        x = self.normalize_layer3(x)
        x = self.activation3(x)
        return x


def train(model: BaseModel, train_loader, epoch=None):
    model.train()
    total_loss = 0.0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(model.device), target.to(model.device)

        # zero the parameter gradients
        model.zero_grad()
        model.optimizer.zero_grad()

        # forward + backward + optimize
        output = model.output(data)
        loss, accuracy = model.loss(output, target)
        model.backward.set_loss(loss)

        model.backward()
        model.optimizer.step()

        # print statistics
        total_loss += loss.item()
        correct += accuracy

        print_mod = int(len(train_loader) / (len(data) * 5))
        if print_mod > 0 and (batch_idx % print_mod == 0 and batch_idx > 0):
            print(
                f'Train Epoch:'
                f' {((epoch + 1) if epoch is not None else "")}'
                f' '
                f'[{batch_idx * len(data)}/{len(train_loader)}'
                f' ({100. * batch_idx / len(train_loader):.0f}%)'
                f']'
                f'\tLoss: {total_loss / (batch_idx * len(data)):.6f}'
            )

    total_loss /= len(train_loader)
    accuracy = correct / len(train_loader)
    return total_loss, accuracy


def accuracy_fn(output, target):
    return exp(-float(
        torch.sum(torch.abs(output - target))
    )) * 100


def main(name, model, parameter_normalize_class, parameter_reduce_precision, parameter_reduce_precision_p):
    if TENSORBOARD:
        model.create_tensorboard(f"{DATA_FOLDER}/{name}")

    model.compile(device=get_device(), layer_data=False)

    model.loss_fn = nn.MSELoss()
    # model.optimizer = ReducePrecisionOptimizer(optim.Adam, model.parameters(), lr=0.01)
    model.accuracy_fn = accuracy_fn
    model.optimizer = optim.Adam(model.parameters(), lr=0.01)

    for i in range(epochs):
        loss, accuracy = train(model, [(data, target)], epochs)
        model.apply_to_parameters(parameter_normalize_class())
        if parameter_reduce_precision_p is None:
            model.apply_to_parameters(parameter_reduce_precision())
        else:
            model.apply_to_parameters(parameter_reduce_precision(precision=parameter_reduce_precision_p))

        if TENSORBOARD:
            model.tensorboard.register_training(i, loss, accuracy)
        if i % int(epochs / 5) == 0:
            print(f"{name.rjust(50)}: {accuracy:.4f}% - loss: {loss:.4f} - output: {model.output(data)} ({i})")

    if bool(torch.any(torch.isnan(model.output(data)))):
        raise ValueError

    if TENSORBOARD:
        model.tensorboard.tensorboard.add_hparams(
            hparam_dict={
                "approach": model.approach.value,
                "std": str(model.std),
                "activation_class": model.activation_class.__name__,
                "output_normalize_class": model.output_normalize_class.__name__,
                "parameter_normalize_class": parameter_normalize_class.__name__,
                "output_reduce_precision": model.output_reduce_precision.__name__,
                "output_reduce_precision_p": str(model.output_reduce_precision_p),
                "parameter_reduce_precision": parameter_reduce_precision.__name__,
                "parameter_reduce_precision_p": str(parameter_reduce_precision_p),
            },
            metric_dict={
                "loss": loss,
                "accuracy": accuracy,
                "accuracy_positive": accuracy_fn(torch.clamp(model.output(data), min=0), torch.clamp(target, min=0)),
                "accuracy_negative": accuracy_fn(torch.clamp(model.output(data), max=0), torch.clamp(target, max=0)),
            }
        )
    save_graph(f"{DATA_FOLDER}/{name}", model(data), model.named_parameters())


def combination_parameters(para: dict, name, list_para, contain=None, sufix=None):
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


def run_approach(approaches):
    parameters = {}
    for approach in approaches:
        parameters[f"{timestamp}_{approach.value}"] = {
            "approach": approach
        }
    parameters = combination_parameters(parameters, "std", [1, 0.1, 0.01], sufix="_" + BackPassTypes.FA.value)
    parameters = combination_parameters(parameters, "std", [1, 0.1, 0.01], sufix="_" + BackPassTypes.DFA.value)
    parameters = combination_parameters(parameters, "std", [1, 0.1, 0.01], sufix="_" + BackPassTypes.RFA.value)
    parameters = combination_parameters(parameters, "std", [1, 0.1, 0.01], sufix="_" + BackPassTypes.RDFA.value)
    parameters = combination_parameters(parameters, "activation_class",
                                        [Identity, LeakyReLU, ReLU, Tanh, ELU, SiLU, GeLU])
    parameters = combination_parameters(parameters, "output_normalize_class", [Clamp, Norm])
    parameters = combination_parameters(parameters, "parameter_normalize_class", [Clamp, Norm])
    parameters = combination_parameters(parameters, "output_reduce_precision", [
        # Identity,
        ReducePrecision,
        # StochasticReducePrecision
    ])
    parameters = combination_parameters(parameters, "output_reduce_precision_p", [2, 4, 6, 8], sufix="_ReducePrecision")
    parameters = combination_parameters(parameters, "output_reduce_precision_p", [2, 4, 6, 8], sufix="_StochasticReducePrecision")
    parameters = combination_parameters(parameters, "parameter_reduce_precision", [
        # Identity,
        ReducePrecision,
        # StochasticReducePrecision
    ])
    parameters = combination_parameters(parameters, "parameter_reduce_precision_p", [2, 4, 6, 8], sufix="_ReducePrecision")
    parameters = combination_parameters(parameters, "parameter_reduce_precision_p", [2, 4, 6, 8], sufix="_StochasticReducePrecision")
    # for k in parameters.keys():
    #     print(k)
    # print()
    total_num = len(list(parameters.keys()))
    print(total_num)
    for i, (name, p) in enumerate(parameters.items()):
        print()
        print(f"{i}/{total_num}")
        model = TestModel(
            approach=p["approach"],
            std=p["std"] if "std" in p else None,
            activation_class=p["activation_class"],
            output_normalize_class=p["output_normalize_class"],
            output_reduce_precision=p["output_reduce_precision"],
            output_reduce_precision_p=p["output_reduce_precision_p"] if "output_reduce_precision_p" in p else None
        )
        main(name, model, p["parameter_normalize_class"], p["parameter_reduce_precision"],
             p["parameter_reduce_precision_p"] if "parameter_reduce_precision_p" in p else None)


if __name__ == '__main__':
    which_approch = BackPassTypes.FA
    DATA_FOLDER = f"C:/_data/tensorboard_{which_approch.value}/"
    set_device("cpu")
    if os.path.exists(DATA_FOLDER):
        shutil.rmtree(DATA_FOLDER)
    os.mkdir(DATA_FOLDER)

    TENSORBOARD = True
    timestamp = str(int(time.time()))
    seed = int(time.time())

    data = (torch.rand((1, 64), device=get_device()) * 2) - 1
    target = torch.Tensor([[-1, -0.50, 0.50, 1]]).to(get_device())
    data /= data.norm()
    print(f"{'target'.rjust(50)}: {target}")

    epochs = 1000
    run_approach([which_approch])
