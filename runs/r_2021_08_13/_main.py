import math
import os
import shutil
import time
from enum import Enum
from math import exp

import torch
from torch import nn, optim

from nn.BaseModel import BaseModel
from nn.TensorboardModelLog import TensorboardModelLog
from nn.activations.Activation import InitImplement
from nn.activations.ELU import ELU
from nn.activations.Gaussian import GeLU
from nn.activations.Identity import Identity
from nn.activations.ReLU import LeakyReLU, ReLU
from nn.activations.SiLU import SiLU
from nn.activations.Tanh import Tanh
from nn.layers.Linear import Linear
from nn.layers.Normalize import Norm, Clamp
from nn.utils.is_using_cuda import get_device, set_device
from nn.utils.make_dot import make_dot


class BackPassTypes(Enum):
    default = "default"
    BP = "bp"
    FA = "FA"
    DFA = "DFA"
    RFA = "RFA"
    RDFA = "RDFA"


def save_graph(filename, output, named_parameters):
    make_dot(output, params=dict(named_parameters)).render(filename + ".svg", cleanup=True)


class NeuralNetwork(nn.Module):
    def __init__(self, activation_class, output_normalize_class):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=64, out_features=16),
            activation_class(),
            output_normalize_class(),
            nn.Linear(in_features=16, out_features=8),
            activation_class(),
            output_normalize_class(),
            nn.Linear(in_features=8, out_features=4),
            activation_class(),
            output_normalize_class(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class TestModel(BaseModel):
    def __init__(
            self,
            approach: BackPassTypes,
            std,
            activation_class,
            output_normalize_class
    ):
        super(TestModel, self).__init__()
        self.approach = approach
        self.linear1 = Linear(in_features=64, out_features=16)
        self.activation1 = activation_class()
        self.normalize_layer1 = output_normalize_class()

        self.linear2 = Linear(in_features=16, out_features=8)
        self.activation2 = activation_class()
        self.normalize_layer2 = output_normalize_class()

        self.linear3 = Linear(in_features=8, out_features=4)
        self.activation3 = activation_class()
        self.normalize_layer3 = output_normalize_class()

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
                self.linear3.backpropagation,
                self.activation2,
                self.normalize_layer2,
                self.linear2.backpropagation,
                self.activation1,
                self.normalize_layer1,
                self.linear1.backpropagation
            )

        if approach == BackPassTypes.FA or approach == BackPassTypes.RFA:
            self.backward.add_relation(
                self.backward.OUTPUT,
                self.activation3,
                self.normalize_layer3,
                self.linear3.feedforward_alignment,
                self.activation2,
                self.normalize_layer2,
                self.linear2.feedforward_alignment,
                self.activation1,
                self.normalize_layer1,
                self.linear1.feedforward_alignment
            )

        if approach == BackPassTypes.DFA or approach == BackPassTypes.RDFA:
            self.backward.add_relation(
                self.backward.OUTPUT,
                self.linear3.direct_feedforward_alignment.pre_backward,
                self.activation3,
                self.normalize_layer3,
                self.linear3.direct_feedforward_alignment
            )
            self.backward.add_relation(
                self.backward.OUTPUT,
                self.linear2.direct_feedforward_alignment.pre_backward,
                self.activation2,
                self.normalize_layer2,
                self.linear2.direct_feedforward_alignment
            )
            self.backward.add_relation(
                self.backward.OUTPUT,
                self.linear1.direct_feedforward_alignment.pre_backward,
                self.activation1,
                self.normalize_layer1,
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
        x = self.normalize_layer1(x)
        x = self.activation1(x)

        x = self.linear2(x)
        x = self.normalize_layer2(x)
        x = self.activation2(x)

        x = self.linear3(x)
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
        loss = model.loss(output, target)
        accuracy = model.accuracy(output, target)
        model.backward.set_loss(loss)
        if bool(torch.any(torch.isnan(output))):
            raise ValueError

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


def main_nn(name, activation_class, parameter_normalize_class, output_normalize_class):
    model = NeuralNetwork(activation_class, output_normalize_class).to(get_device())

    if TENSORBOARD:
        tensorboard = TensorboardModelLog(model, f"{DATA_FOLDER}/{name}")

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for i in range(epochs):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        output = model(data)
        loss = loss_fn(output, target)
        if TENSORBOARD:
            tensorboard.register_training(i, loss.item(), accuracy_fn(output, target), layer_data=False)

        loss.backward()
        optimizer.step()
        BaseModel.apply_to_parameters(model, parameter_normalize_class())

        if i % int(epochs / 5) == 0:
            print(
                f"{name.rjust(50)}: {accuracy_fn(output, target):.4f}% - loss: {loss:.4f} - output: {model(data)} ({i})")
    if TENSORBOARD:
        tensorboard.tensorboard.add_hparams(
            hparam_dict={
                "approach": "nn",
                "std": "None",
                "activation_class": activation_class.__name__,
                "parameter_normalize_class": parameter_normalize_class.__name__,
                "output_normalize_class": output_normalize_class.__name__,
            },
            metric_dict={
                "loss": loss.item(),
                "accuracy": accuracy_fn(output, target),
                "accuracy_positive": accuracy_fn(torch.clamp(output, min=0), torch.clamp(target, min=0)),
                "accuracy_negative": accuracy_fn(torch.clamp(output, max=0), torch.clamp(target, max=0)),
            },
        )
    save_graph(f"{DATA_FOLDER}/{name}", model(data), model.named_parameters())


def main(name, approach, std, activation_class, parameter_normalize_class, output_normalize_class):
    model = TestModel(approach, std, activation_class, output_normalize_class)
    if TENSORBOARD:
        model.create_tensorboard(f"{DATA_FOLDER}/{name}")

    model.compile(device=get_device(), layer_data=False)

    model.loss = nn.MSELoss()
    model.optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.accuracy = accuracy_fn

    for i in range(epochs):
        loss, accuracy = train(model, [(data, target)], epochs)
        if TENSORBOARD:
            model.tensorboard.register_training(i, loss, accuracy, layer_data=False)
        model.apply_to_parameters(parameter_normalize_class())

        # if accuracy > 99.5:
        #     break

        if i % int(epochs / 5) == 0:
            print(f"{name.rjust(50)}: {accuracy:.4f}% - loss: {loss:.4f} - output: {model.output(data)} ({i})")

    if TENSORBOARD:
        model.tensorboard.tensorboard.add_hparams(
            hparam_dict={
                "approach": approach.value,
                "std": str(std),
                "activation_class": activation_class.__name__,
                "parameter_normalize_class": parameter_normalize_class.__name__,
                "output_normalize_class": output_normalize_class.__name__,
            },
            metric_dict={
                "loss": loss,
                "accuracy": accuracy,
                "accuracy_positive": accuracy_fn(torch.clamp(model.output(data), min=0), torch.clamp(target, min=0)),
                "accuracy_negative": accuracy_fn(torch.clamp(model.output(data), max=0), torch.clamp(target, max=0)),
            }
        )
    save_graph(f"{DATA_FOLDER}/{name}", model(data), model.named_parameters())


if __name__ == '__main__':
    DATA_FOLDER = f"C:/_data/tensorboard/"
    set_device("cpu")
    if os.path.exists(DATA_FOLDER):
        shutil.rmtree(DATA_FOLDER)
    os.mkdir(DATA_FOLDER)

    TENSORBOARD = True
    data = (torch.rand((1, 64), device=get_device()) * 2) - 1
    target = torch.Tensor([[-1, -0.50, 0.50, 1]]).to(get_device())
    data /= data.norm()
    target /= target.norm()
    print(f"{'target'.rjust(50)}: {target}")

    for i in range(10):
        epochs = 1000
        timestamp = str(int(time.time()))
        seed = int(time.time())
        for ac in [
            Identity,
            LeakyReLU,
            ReLU,
            Tanh,
            ELU,
            SiLU,
            GeLU,
        ]:
            for nfn in [
                Clamp,
                Norm,
            ]:
                for onl in [
                    Clamp,
                    Norm,
                ]:
                    # torch.manual_seed(seed)
                    # main_nn(f"{timestamp}_nn_{ac.__name__}_{nfn.__name__}_{onl.__name__}", activation_class=ac, parameter_normalize_class=nfn, output_normalize_class=onl)
                    # for app in [
                    #     BackPassTypes.default,
                    #     BackPassTypes.BP
                    # ]:
                    #     torch.manual_seed(seed)
                    #     main(f"{timestamp}_{app.value}_{ac.__name__}_{nfn.__name__}_{onl.__name__}", approach=app, std=None, activation_class=ac, parameter_normalize_class=nfn, output_normalize_class=onl)

                    for app in [
                        # BackPassTypes.FA,
                        # BackPassTypes.RFA,
                        BackPassTypes.DFA,
                        BackPassTypes.RDFA
                    ]:
                        for s in [
                            1,
                            0.1,
                            0.01,
                            0.001,
                        ]:
                            torch.manual_seed(seed)
                            main(f"{timestamp}_{app.value}_{s}_{ac.__name__}_{nfn.__name__}_{onl.__name__}",
                                 approach=app, std=s, activation_class=ac, parameter_normalize_class=nfn,
                                 output_normalize_class=onl)
