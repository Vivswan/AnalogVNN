import torch
from torch import nn, Tensor

from nn.BaseLayer import BaseLayer
from nn.backward_pass.BackwardFunction import BackwardIdentity
from nn.fn.reduce_precision import stochastic_reduce_precision
from nn.parameters.StochasticReducePrecisionParameter import StochasticReducePrecisionParameter


class StochasticReducePrecision(BaseLayer, BackwardIdentity):
    __constants__ = ['precision']
    precision: nn.Parameter
    parameter_class = StochasticReducePrecisionParameter

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

    def forward(self, x: Tensor):
        return stochastic_reduce_precision(x, self.precision)
