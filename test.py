from enum import Enum

import torch
from torch import nn, optim

from nn.Sequential import Sequential
from nn.layers.linear import Linear
from nn.activations.relu import ReLU
from nn.utils.make_dot import make_dot


class BackPassTypes(Enum):
    default = "default"
    BP = "bp"
    FA = "FA"
    DFA = "DFA"
    RFA = "RFA"
    RDFA = "RDFA"


def save_graph(filename, output, named_parameters):
    make_dot(output, params=dict(named_parameters)).render(filename, cleanup=True)


def main(approach, std=None):
    torch.manual_seed(run)
    layer_1 = Linear(in_features=8, out_features=4)
    relu_1 = ReLU()
    layer_2 = Linear(in_features=4, out_features=2)
    relu_2 = ReLU()

    model = Sequential(
        layer_1,
        relu_1,
        layer_2,
        relu_2,
    )

    if approach == BackPassTypes.default:
        model.backward.use_default_graph = True

    # Backpropagation
    if approach == BackPassTypes.BP:
        model.backward.add_relation(model.backward.OUTPUT, relu_2, layer_2.backpropagation, relu_1,
                                    layer_1.backpropagation)

    # Feedback Alignment
    if approach == BackPassTypes.FA or approach == BackPassTypes.RFA:
        model.backward.add_relation(model.backward.OUTPUT, relu_2, layer_2.feedforward_alignment, relu_1,
                                    layer_1.feedforward_alignment)

    # Direct Feedback Alignment
    if approach == BackPassTypes.DFA or approach == BackPassTypes.RDFA:
        model.backward.add_relation(model.backward.OUTPUT, relu_2, layer_2.direct_feedforward_alignment)
        model.backward.add_relation(model.backward.OUTPUT, relu_1, layer_1.direct_feedforward_alignment)

    if approach == BackPassTypes.RFA:
        layer_1.feedforward_alignment.is_fixed = False
        layer_2.feedforward_alignment.is_fixed = False

    if approach == BackPassTypes.RDFA:
        layer_1.direct_feedforward_alignment.is_fixed = False
        layer_2.direct_feedforward_alignment.is_fixed = False

    if std is not None:
        layer_1.feedforward_alignment.std = std
        layer_2.feedforward_alignment.std = std
        layer_1.direct_feedforward_alignment.std = std
        layer_2.direct_feedforward_alignment.std = std

    name = f"{str(run)}_{approach.value}{'' if std is None else f'-{std:.3f}'}"
    # model.create_tensorboard(f"D:/_data/tensorboard/{name}")
    model.compile()

    data = torch.rand((1, 8))
    target = torch.rand((1, 2))
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for i in range(epochs):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        output = model.output(data)
        loss = loss_fn(output, target)
        model.backward.set_loss(loss)
        # model.tensorboard.register_training(i, 1 - exp(-abs(loss.item())), exp(-abs(float(torch.sum(output) - torch.sum(target)))) * 100)

        model.backward()
        optimizer.step()

        print(f"{name}: {loss} - {target} - {model.output(data)}")

    if approach == BackPassTypes.default:
        save_graph("test", model.output(data), model.named_parameters())


if __name__ == '__main__':
    epochs = 5
    for run in range(1):
        main(BackPassTypes.default)
        main(BackPassTypes.BP)
        # main(BackPassTypes.FA, std=1)
        # main(BackPassTypes.FA, std=0.1)
        # main(BackPassTypes.FA, std=0.01)
        # main(BackPassTypes.RFA, std=1)
        # main(BackPassTypes.RFA, std=0.1)
        # main(BackPassTypes.RFA, std=0.01)
        # main(BackPassTypes.DFA, std=1)
        # main(BackPassTypes.DFA, std=0.1)
        # main(BackPassTypes.DFA, std=0.01)
        # main(BackPassTypes.RDFA, std=1)
        # main(BackPassTypes.RDFA, std=0.1)
        # main(BackPassTypes.RDFA, std=0.01)
    data = torch.Tensor([[2, 3]])
    print(data / data)
