import torch
from torch import nn, Tensor

from nn.BaseLayer import BaseLayer
from nn.backward_pass.BackwardFunction import BackwardFunction


def stochastic_reduce_precision(x: Tensor, precision: int) -> Tensor:
    g: Tensor = x * precision
    rand_x = torch.rand_like(g, requires_grad=False)

    g_abs = torch.abs(g)
    g_floor = torch.floor(g_abs)
    g_ceil = torch.ceil(g_abs)

    prob_floor = 1 - torch.abs(g_floor - g_abs)
    bool_floor = rand_x <= prob_floor
    do_floor = bool_floor.type(torch.float)
    do_ceil = torch.logical_not(bool_floor).type(torch.float)

    f = torch.sign(g) * (do_floor * g_floor + do_ceil * g_ceil) * (1 / precision)
    return f


class StochasticReducePrecision(BaseLayer, BackwardFunction):
    __constants__ = ['precision']
    precision: nn.Parameter

    def __init__(self, precision: int = 8):
        super(StochasticReducePrecision, self).__init__()
        if precision < 1:
            raise ValueError("precision has to be more than 0, but got {}".format(precision))

        if precision != int(precision):
            raise ValueError("precision must be int, but got {}".format(precision))

        self.precision = nn.Parameter(torch.tensor(precision), requires_grad=False)

    @property
    def step_size(self) -> float:
        return 1 / self.precision

    def extra_repr(self) -> str:
        return f'precision={self.precision}'

    def forward(self, x: Tensor, force=False):
        return stochastic_reduce_precision(x, self.precision.data)
