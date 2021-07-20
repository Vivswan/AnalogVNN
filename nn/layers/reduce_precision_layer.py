import torch
from torch import nn, Tensor


class ReducePrecision(nn.Module):
    __constants__ = ['precision', 'divide']
    precision: int
    divide: float

    def __init__(self, precision: int = 8, divide: float = 0.5):
        super(ReducePrecision, self).__init__()
        if precision < 1:
            raise ValueError("precision has to be more than 0, but got {}".format(precision))

        if precision != int(precision):
            raise ValueError("precision must be int, but got {}".format(precision))

        if not (0 <= divide <= 1):
            raise ValueError("divide must be between 0 and 1, but got {}".format(divide))

        self.precision = precision
        self.divide = divide

    def extra_repr(self) -> str:
        return f'precision={self.precision}, divide={self.divide}'

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            return torch.sign(input) * torch.max(
                torch.floor(torch.abs(input) * self.precision),
                torch.ceil(torch.abs(input) * self.precision - self.divide),
            ) * 1 / self.precision
        return input


if __name__ == '__main__':
    input = torch.Tensor([1.9])
    print(f"input: {input}")
    print(f"p = 2: {ReducePrecision(precision=2)(input)}")
    print(f"p = 4: {ReducePrecision(precision=4)(input)}")
    print(f"p = 8: {ReducePrecision(precision=8)(input)}")
