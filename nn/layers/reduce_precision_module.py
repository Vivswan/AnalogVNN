import torch
from torch import Tensor, nn

from nn.layers.reduce_precision_layer import ReducePrecision


class ReducePrecisionModule(nn.Module):
    def __init__(self, module: nn.Module, precision: int = 8, divide: float = 0.5) -> None:
        super(ReducePrecisionModule, self).__init__()
        self.parameter_precision = ReducePrecision(precision=precision, divide=divide)
        self.module = module

    def forward(self, input: Tensor) -> Tensor:
        return self.module(input)

    def step(self):
        with torch.no_grad():
            for parameter in self.module.parameters():
                parameter.data.copy_(self.parameter_precision(parameter.data))


if __name__ == '__main__':
    print()
    torch.manual_seed(0)

    model = ReducePrecisionModule(nn.Linear(1, 1), precision=8)
    input = torch.Tensor([1])
    target = torch.Tensor([1])

    with torch.no_grad():
        for p in model.parameters():
            p.data = torch.zeros_like(p.data)

        print(f"output before: {model(input)}")
        for p in model.named_parameters():
            print(f"p before  - {p[0]} ({p[1].requires_grad}): {p[1].data}")

    loss_fn = nn.MSELoss()

    print()
    for i in range(20):
        model.train()

        output = model(input)
        loss = loss_fn(output, target)

        # print(f"loss ({i}): {loss.item()}")

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                # print(f"grad: {p.grad}")
                p.copy_(p - 0.1 * p.grad)

    print()
    with torch.no_grad():
        for p in model.named_parameters():
            print(f"p after  - {p[0]} ({p[1].requires_grad}): {p[1].data}")

        print(f"output after : {model(input)}")
