import os
import shutil
import time
from math import exp

import torch
from torch import nn, optim

from nn.TensorboardModelLog import TensorboardModelLog
from nn.activations.relu import ReLU, LeakyReLU
from nn.layers.linear import Linear
from nn.model_base import BaseModel
from nn.utils.is_using_cuda import get_device
from nn.utils.make_dot import make_dot
from runs.r_2021_07_31._apporaches import BackPassTypes


def save_graph(filename, output, named_parameters):
    make_dot(output, params=dict(named_parameters)).render(filename, cleanup=True)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU() if not leaky_relu else nn.LeakyReLU(),
            nn.Linear(in_features=16, out_features=4),
            nn.ReLU() if not leaky_relu else nn.LeakyReLU(),
            nn.Linear(in_features=4, out_features=1),
            nn.ReLU() if not leaky_relu else nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class TestModel(BaseModel):
    def __init__(
            self,
            approach: BackPassTypes,
            std=1
    ):
        super(TestModel, self).__init__()
        self.approach = approach
        self.linear1 = Linear(in_features=64, out_features=16, activation=ReLU() if not leaky_relu else LeakyReLU())
        self.linear2 = Linear(in_features=16, out_features=4, activation=ReLU() if not leaky_relu else LeakyReLU())
        self.linear3 = Linear(in_features=4, out_features=1, activation=ReLU() if not leaky_relu else LeakyReLU())

        if self.approach == BackPassTypes.default:
            self.backward.use_default_graph = True

        if approach == BackPassTypes.default:
            self.backward.use_default_graph = True

        if approach == BackPassTypes.BP:
            self.backward.add_relation(self.backward.OUTPUT,
                                       self.linear3.backpropagation,
                                       self.linear2.backpropagation,
                                       self.linear1.backpropagation)

        if approach == BackPassTypes.FA or approach == BackPassTypes.RFA:
            self.backward.add_relation(self.backward.OUTPUT,
                                       self.linear3.feedforward_alignment,
                                       self.linear2.feedforward_alignment,
                                       self.linear1.feedforward_alignment)

        if approach == BackPassTypes.DFA or approach == BackPassTypes.RDFA:
            self.backward.add_relation(self.backward.OUTPUT, self.linear3.direct_feedforward_alignment)
            self.backward.add_relation(self.backward.OUTPUT, self.linear2.direct_feedforward_alignment)
            self.backward.add_relation(self.backward.OUTPUT, self.linear1.direct_feedforward_alignment)

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
        x = self.linear2(x)
        x = self.linear3(x)
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


def main_nn():
    # print()
    torch.manual_seed(run)

    model = NeuralNetwork().to(get_device())

    name = f"{timestamp}{run}_nn"
    if TENSORBOARD:
        tensorboard = TensorboardModelLog(model, f"{DATA_FOLDER}/{name}")

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    accuracy = lambda x: exp(-float(torch.sum(torch.abs(x - target)))) * 100

    for i in range(epochs):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        output = model(data)
        loss = loss_fn(output, target)
        if TENSORBOARD:
            with torch.no_grad():
                tensorboard.register_training(i, loss.item(), accuracy(output))

        loss.backward()
        optimizer.step()
        # model.normalize()

        if epochs <= 5 or i == epochs - 1:
            print(f"{name.rjust(20)}: {accuracy(output):.4f}% - loss: {loss:.4f} - output: {model(data)}")
    save_graph(f"{DATA_FOLDER}/nn", model(data), model.named_parameters())


def main(approach, std=None):
    # print()
    torch.manual_seed(run)

    name = f"{timestamp}{run}_{approach.value}{'' if std is None else f'-{std:.2f}'}"
    model = TestModel(approach, std)
    if TENSORBOARD:
        model.create_tensorboard(f"{DATA_FOLDER}/{name}")

    model.compile(device=get_device())

    model.loss = nn.MSELoss()
    model.optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.accuracy = lambda o, t: exp(-abs(float(torch.sum(o - t)))) * 100

    for i in range(epochs):
        loss, accuracy = train(model, [(data, target)], epochs)
        if TENSORBOARD:
            model.tensorboard.register_training(i, loss, accuracy)
        # model.normalize()

        if epochs <= 5 or i == epochs - 1:
            print(f"{name.rjust(20)}: {accuracy:.4f}% - loss: {loss:.4f} - output: {model.output(data)}")
    save_graph(f"{DATA_FOLDER}/{approach.value}", model(data), model.named_parameters())


if __name__ == '__main__':
    DATA_FOLDER = "D:/_data/tensorboard2/"
    # shutil.rmtree(DATA_FOLDER)
    # os.mkdir(DATA_FOLDER)
    TENSORBOARD = False
    # timestamp = ""
    timestamp = str(int(time.time()))

    data = (torch.rand((1, 64), device=get_device()) * 2) - 1
    target = torch.Tensor([[0.5]]).to(get_device())
    print(f"{'target'.rjust(20)}: {target}")

    leaky_relu = True
    epochs = 5
    for run in range(1):
        # main_nn()
        main(BackPassTypes.default)
        # main(BackPassTypes.BP)
        # main(BackPassTypes.FA, std=1)
        # main(BackPassTypes.FA, std=0.1)
        # main(BackPassTypes.FA, std=0.01)
        # main(BackPassTypes.RFA, std=1)
        # main(BackPassTypes.RFA, std=0.1)
        # main(BackPassTypes.RFA, std=0.01)
        main(BackPassTypes.DFA, std=1)
        # main(BackPassTypes.DFA, std=0.1)
        # main(BackPassTypes.DFA, std=0.01)
        # main(BackPassTypes.RDFA, std=1)
        # main(BackPassTypes.RDFA, std=0.1)
        # main(BackPassTypes.RDFA, std=0.01)
