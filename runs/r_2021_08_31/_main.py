import os
import shutil
import time
from math import exp

import torch
from torch import nn, optim

from nn.TensorboardModelLog import TensorboardModelLog
from nn.activations.elu import ELU
from nn.activations.gaussian import GeLU
from nn.activations.identity import Identity
from nn.activations.relu import ReLU, LeakyReLU
from nn.activations.sigmoid import Tanh, SiLU
from nn.layers.linear import Linear
from nn.layers.normalize import Normalize, Clamp
from nn.model_base import BaseModel
from nn.utils.is_using_cuda import get_device
from nn.utils.make_dot import make_dot
from nn.utils.normalize import normalize_model, normalize_cutoff_model
from runs.r_2021_08_31._apporaches import BackPassTypes


def save_graph(filename, output, named_parameters):
    make_dot(output, params=dict(named_parameters)).render(filename + ".svg", cleanup=True)


class NeuralNetwork(nn.Module):
    def __init__(self, activation_class, output_normalize_layer):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=64, out_features=16),
            activation_class(),
            output_normalize_layer(),
            nn.Linear(in_features=16, out_features=4),
            activation_class(),
            output_normalize_layer(),
            nn.Linear(in_features=4, out_features=2),
            activation_class(),
            output_normalize_layer(),
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
            output_normalize_layer
    ):
        super(TestModel, self).__init__()
        self.approach = approach
        self.linear1 = Linear(in_features=64, out_features=16, activation=activation_class())
        self.output_normalize_layer1 = output_normalize_layer()
        self.linear2 = Linear(in_features=16, out_features=4, activation=activation_class())
        self.output_normalize_layer2 = output_normalize_layer()
        self.linear3 = Linear(in_features=4, out_features=2, activation=activation_class())
        self.output_normalize_layer3 = output_normalize_layer()

        if self.approach == BackPassTypes.default:
            self.backward.use_default_graph = True

        if approach == BackPassTypes.default:
            self.backward.use_default_graph = True

        if approach == BackPassTypes.BP:
            self.backward.add_relation(self.backward.OUTPUT,
                                       self.output_normalize_layer3,
                                       self.linear3.backpropagation,
                                       self.output_normalize_layer2,
                                       self.linear2.backpropagation,
                                       self.output_normalize_layer1,
                                       self.linear1.backpropagation)

        if approach == BackPassTypes.FA or approach == BackPassTypes.RFA:
            self.backward.add_relation(self.backward.OUTPUT,
                                       self.output_normalize_layer3,
                                       self.linear3.feedforward_alignment,
                                       self.output_normalize_layer2,
                                       self.linear2.feedforward_alignment,
                                       self.output_normalize_layer1,
                                       self.linear1.feedforward_alignment)

        if approach == BackPassTypes.DFA or approach == BackPassTypes.RDFA:
            self.backward.add_relation(self.backward.OUTPUT, self.output_normalize_layer3, self.linear3.direct_feedforward_alignment)
            self.backward.add_relation(self.backward.OUTPUT, self.output_normalize_layer2, self.linear2.direct_feedforward_alignment)
            self.backward.add_relation(self.backward.OUTPUT, self.output_normalize_layer1, self.linear1.direct_feedforward_alignment)

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
        x = self.output_normalize_layer1(x)
        x = self.linear2(x)
        x = self.output_normalize_layer2(x)
        x = self.linear3(x)
        x = self.output_normalize_layer3(x)
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


def main_nn(name, activation_class, normalize_fn, output_normalize_layer):
    model = NeuralNetwork(activation_class, output_normalize_layer).to(get_device())

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
        if normalize_fn is not None:
            normalize_fn(model)

        if i % int(epochs / 5) == 0:
            print(f"{name.rjust(40)}: {accuracy_fn(output, target):.4f}% - loss: {loss:.4f} - output: {model(data)} ({i})")
    if TENSORBOARD:
        tensorboard.tensorboard.add_hparams(
            hparam_dict={
                "approach": "nn",
                "std": -1,
                "activation_class": activation_class.__name__,
                "normalize_fn": "None" if normalize_fn is None else normalize_fn.__name__,
                "output_normalize_layer": output_normalize_layer.__name__,
            },
            metric_dict={
                "loss": loss.item(),
                "accuracy": accuracy_fn(output, target),
                f"output_negative {target[0][0]:.2f}": model(data)[0][0],
                f"output_positive {target[0][1]:.2f}": model(data)[0][1],
            },
        )
    save_graph(f"{DATA_FOLDER}/{name}", model(data), model.named_parameters())


def main(name, approach, std, activation_class, normalize_fn, output_normalize_layer):
    model = TestModel(approach, std, activation_class, output_normalize_layer)
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
        if normalize_fn is not None:
            normalize_fn(model)

        if i % int(epochs / 5) == 0:
            print(f"{name.rjust(40)}: {accuracy:.4f}% - loss: {loss:.4f} - output: {model.output(data)} ({i})")
    if TENSORBOARD:
        model.tensorboard.tensorboard.add_hparams(
            hparam_dict={
                "approach": approach.value,
                "std": -1 if std is None else std,
                "activation_class": activation_class.__name__,
                "normalize_fn": "None" if normalize_fn is None else normalize_fn.__name__,
                "output_normalize_layer": output_normalize_layer.__name__,
            },
            metric_dict={
                "loss": loss,
                "accuracy": accuracy,
                f"output_negative {target[0][0]:.2f}": model.output(data)[0][0],
                f"output_positive {target[0][1]:.2f}": model.output(data)[0][1],
            }
        )
    save_graph(f"{DATA_FOLDER}/{name}", model(data), model.named_parameters())


if __name__ == '__main__':
    DATA_FOLDER = "C:/_data/tensorboard/"
    if os.path.exists(DATA_FOLDER):
        shutil.rmtree(DATA_FOLDER)
    os.mkdir(DATA_FOLDER)
    TENSORBOARD = True
    timestamp = str(int(time.time()))
    seed = int(time.time())

    data = (torch.rand((1, 64), device=get_device()) * 2) - 1
    target = torch.Tensor([[-0.75, 0.75]]).to(get_device())
    data /= data.norm()
    target /= target.norm()
    print(f"{'target'.rjust(40)}: {target}")

    epochs = 1000
    for ac in [
        Identity,
        ReLU,
        LeakyReLU,
        Tanh,
        ELU,
        SiLU,
        GeLU,
    ]:
        for nfn in [
            # None,
            normalize_cutoff_model,
            normalize_model,
        ]:
            if nfn == normalize_model:
                nfn_str = "n"
            elif nfn == normalize_cutoff_model:
                nfn_str = "c"
            else:
                nfn_str = ""

            for onl in [
                Clamp,
                Identity,
                Normalize,
            ]:
                torch.manual_seed(seed)
                timestamp = str(int(time.time()))
                main_nn(f"{timestamp}_nn_{ac.__name__}_{nfn_str}_{onl.__name__}", activation_class=ac, normalize_fn=nfn, output_normalize_layer=onl)
                for app in [
                    BackPassTypes.default,
                    BackPassTypes.BP
                ]:
                    torch.manual_seed(seed)
                    timestamp = str(int(time.time()))
                    main(f"{timestamp}_{app.value}_{ac.__name__}_{nfn_str}_{onl.__name__}", approach=app, std=None, activation_class=ac, normalize_fn=nfn, output_normalize_layer=onl)

                for app in [
                    BackPassTypes.FA,
                    BackPassTypes.RFA,
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
                        timestamp = str(int(time.time()))
                        main(f"{timestamp}_{app.value}_{s}_{ac.__name__}_{nfn_str}_{onl.__name__}", approach=app, std=s, activation_class=ac, normalize_fn=nfn, output_normalize_layer=onl)
