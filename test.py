from enum import Enum

import torch
from torch import nn, optim

from nn.Sequential import Sequential
from nn.layers.linear import Linear
from nn.utils.make_dot import make_dot

epochs = 100

class BackPassTypes(Enum):
    BP = "bp"
    FA = "FA"
    DFA = "DFA"
    RFA = "RFA"
    RDFA = "RDFA"

def save_graph(filename, output, named_parameters):
    make_dot(output, params=dict(named_parameters)).render(filename, cleanup=True)

def main1():
    torch.manual_seed(0)
    layer_1 = Linear(in_features=3, out_features=2)
    layer_2 = Linear(in_features=2, out_features=1)

    model = Sequential(
        layer_1,
        layer_2,
    )

    data = torch.ones((3,))
    target = torch.ones((1,))
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for i in range(epochs):
        model.train()
        model.zero_grad()

        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # print(f"loss ({i}): {loss.item()}")
    print(f"loss: {loss.item()}")
    print(f"result: {model(data)}")


def main2(approach):
    torch.manual_seed(0)
    layer_1 = Linear(in_features=3, out_features=2)
    layer_2 = Linear(in_features=2, out_features=1)

    model = Sequential(
        layer_1,
        layer_2,
    )

    # Backpropagation
    if approach == BackPassTypes.BP:
        model.backward.add_relation(model.backward.OUTPUT, layer_2.backpropagation)
        model.backward.add_relation(layer_2.backpropagation, layer_1.backpropagation)

    # Feedback Alignment
    if approach == BackPassTypes.FA or approach == BackPassTypes.RFA:
        model.backward.add_relation(model.backward.OUTPUT, layer_2.feedforward_alignment)
        model.backward.add_relation(layer_2.feedforward_alignment, layer_1.feedforward_alignment)

    # Direct Feedback Alignment
    if approach == BackPassTypes.DFA or approach == BackPassTypes.RDFA:
        model.backward.add_relation(model.backward.OUTPUT, layer_2.feedforward_alignment)
        model.backward.add_relation(model.backward.OUTPUT, layer_1.feedforward_alignment)

    if approach == BackPassTypes.RFA or approach == BackPassTypes.RDFA:
        layer_1.set_feedforward_alignment_params(std=0.01, is_fixed=False)
        layer_2.set_feedforward_alignment_params(std=0.01, is_fixed=False)

    model.compile()

    data = torch.ones((3,))
    target = torch.ones((1,))
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for i in range(epochs):
        model.train()

        output = model.output(data)
        loss = loss_fn(output, target)
        model.backward.set_loss(loss)

        model.backward()
        optimizer.step()

        # print(f"loss ({i}): {loss.item()}")
    print(f"loss: {loss.item()}")
    print(f"result: {model.output(data)}")

if __name__ == '__main__':
    print("run normal")
    main1()

    print()
    print("run BP")
    main2(BackPassTypes.BP)
    print()
    print("run FA")
    main2(BackPassTypes.FA)
    print()
    print("run RFA")
    main2(BackPassTypes.RFA)
    print()
    print("run DFA")
    main2(BackPassTypes.DFA)
    print()
    print("run RDFA")
    main2(BackPassTypes.RDFA)
