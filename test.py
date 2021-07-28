import torch
from torch import Tensor

from nn.Sequential import Sequential
from nn.layers.linear import Linear
from nn.utils.make_dot import make_dot

def save_graph(filename, output, named_parameters):
    make_dot(output, params=dict(named_parameters)).render(filename, cleanup=True)

def main1():
    torch.manual_seed(0)
    layer_1 = Linear(in_features=1, out_features=1)
    layer_2 = Linear(in_features=1, out_features=1)

    model = Sequential(
        layer_1,
        layer_2,
    )


    data = torch.ones((1,))
    target = torch.ones((1,))


    for i in range(5):
        model.train()
        model.zero_grad()

        output = model(data)
        output.rename_("output")
        loss = torch.mean((output - target) ** 2)

        if i == 0:
            save_graph("main1", loss, model.named_parameters())

        print(f"loss ({i}): {loss.item()}")

        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.copy_(p - 0.1 * p.grad)


def main2():
    torch.manual_seed(0)
    layer_1 = Linear(in_features=1, out_features=1)
    layer_2 = Linear(in_features=1, out_features=1)

    model = Sequential(
        layer_1,
        layer_2,
    )
    model.compile()

    model.backward.to(layer_2)
    layer_2.backward.to(layer_1)

    data = torch.ones((1,))
    target = torch.ones((1,))

    for i in range(5):
        model.train()
        model.zero_grad()

        output: Tensor = model(data)
        output.rename_("output")
        model.backward.set_output(output)

        loss = torch.mean((output - target) ** 2)
        model.backward.set_loss(loss)
        print(f"loss ({i}): {loss.item()}")

        model.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.copy_(p - 0.1 * p.grad)


if __name__ == '__main__':
    print("run 1")
    main1()

    print()
    print("run 2")
    main2()
