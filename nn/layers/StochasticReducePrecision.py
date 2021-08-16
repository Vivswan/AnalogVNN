import torch
from torch import nn, Tensor

from nn.BackwardFunction import BackwardFunction
from nn.BaseLayer import BaseLayer


class StochasticReducePrecision(BaseLayer, BackwardFunction):
    __constants__ = ['precision', 'divide']
    precision: nn.Parameter
    divide: nn.Parameter

    def __init__(self, precision: int = 8, divide: float = 0.5):
        super(StochasticReducePrecision, self).__init__()
        if precision < 1:
            raise ValueError("precision has to be more than 0, but got {}".format(precision))

        if precision != int(precision):
            raise ValueError("precision must be int, but got {}".format(precision))

        self.precision = nn.Parameter(torch.tensor(precision), requires_grad=False)
        self.divide = nn.Parameter(torch.tensor(divide), requires_grad=False)

    @property
    def step_size(self) -> float:
        return 1 / self.precision

    def extra_repr(self) -> str:
        return f'precision={self.precision}, divide={self.divide}'

    def forward(self, x: Tensor):
        if self.training:
            g: Tensor = x * self.precision
            rand_x = torch.rand_like(g, requires_grad=False)

            g_abs = torch.abs(g)
            g_floor = torch.floor(g_abs)
            g_ceil = torch.ceil(g_abs)

            prob_floor = 1 - torch.abs(g_floor - g_abs)
            bool_floor = rand_x <= prob_floor
            do_floor = bool_floor.type(torch.float)
            do_ceil = torch.logical_not(bool_floor).type(torch.float)

            f = torch.sign(g) * (do_floor * g_floor + do_ceil * g_ceil) * (1 / self.precision)
            return f
        else:
            return x

    def backward(self, grad_output: Tensor):
        return grad_output
