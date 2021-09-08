import torch
from torch import nn, Tensor

from nn.BaseLayer import BaseLayer
from nn.backward_pass.BackwardFunction import BackwardFunction


class ReducePrecision(BaseLayer, BackwardFunction):
    __constants__ = ['precision', 'divide', 'shift']
    precision: nn.Parameter
    divide: nn.Parameter
    shift: nn.Parameter

    def __init__(self, precision: int = 8, divide: float = 0.5, shift: float = 0):
        super(ReducePrecision, self).__init__()
        if precision < 1:
            raise ValueError("precision has to be more than 0, but got {}".format(precision))

        if precision != int(precision):
            raise ValueError("precision must be int, but got {}".format(precision))

        if not (0 <= divide <= 1):
            raise ValueError("divide must be between 0 and 1, but got {}".format(divide))

        if not (-1 <= shift <= 1):
            raise ValueError("shift must be between -1 and 1, but got {}".format(shift))

        self.precision = nn.Parameter(torch.tensor(precision), requires_grad=False)
        self.divide = nn.Parameter(torch.tensor(divide), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(shift), requires_grad=False)

    @property
    def step_size(self) -> float:
        return 1 / self.precision

    def extra_repr(self) -> str:
        return f'precision={self.precision}, divide={self.divide}, shift={self.shift}'

    def forward(self, x: Tensor, force=False):
        if self.training or force:
            g: Tensor = x * self.precision
            f = torch.sign(g) * torch.maximum(
                torch.floor(torch.abs(g)),
                torch.ceil(torch.abs(g) - self.divide)
            ) * (1 / self.precision)
            return f
        else:
            return x

    def backward(self, grad_output: Tensor):
        return self.forward(grad_output, force=True)
