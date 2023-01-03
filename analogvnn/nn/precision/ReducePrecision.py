import torch
from torch import nn, Tensor

from analogvnn.backward.BackwardIdentity import BackwardIdentity
from analogvnn.fn.reduce_precision import reduce_precision
from analogvnn.nn.module.Layer import Layer


class ReducePrecision(Layer, BackwardIdentity):
    __constants__ = ['precision', 'divide']
    precision: nn.Parameter
    divide: nn.Parameter

    def __init__(self, precision: int = None, divide: float = 0.5):
        super(ReducePrecision, self).__init__()
        if precision < 1:
            raise ValueError(f"precision has to be more than 0, but got {precision}")

        if precision != int(precision):
            raise ValueError(f"precision must be int, but got {precision}")

        if not (0 <= divide <= 1):
            raise ValueError(f"divide must be between 0 and 1, but got {divide}")

        self.precision = nn.Parameter(torch.tensor(precision), requires_grad=False)
        self.divide = nn.Parameter(torch.tensor(divide), requires_grad=False)

    @property
    def step_width(self) -> float:
        return 1 / self.precision

    @property
    def bit_precision(self):
        return torch.log2(self.precision + 1)

    @staticmethod
    def get_precision(bit_precision):
        return 2 ** bit_precision - 1

    def extra_repr(self) -> str:
        return f'precision={int(self.precision)}, divide={float(self.divide):0.2f}'

    def forward(self, x: Tensor):
        return reduce_precision(x, self.precision, self.divide)
