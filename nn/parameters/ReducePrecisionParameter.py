import torch

from nn.layers.ReducePrecision import reduce_precision
from nn.parameters.BasePrecisionParameter import BasePrecisionParameter


class ReducePrecisionParameter(BasePrecisionParameter):
    precision: int
    divide: float

    def __init__(self, data=None, requires_grad=True, precision: int = 8, divide: float = 0.5,
                 use_zero_pseudo_tensor=False):
        self.precision = precision
        self.divide = divide
        super(ReducePrecisionParameter, self).__init__(data, requires_grad, use_zero_pseudo_tensor)

    def __repr__(self):
        return f'ReducePrecisionParameter(precision: {self.precision}, divide:{self.divide}): ' + super(
            ReducePrecisionParameter, self).__repr__()

    @torch.no_grad()
    def set_tensor(self, data, precision=None, divide=None):
        if precision is None:
            precision = self.precision
        if divide is None:
            divide = self.divide
        self.data = reduce_precision(data, precision=precision, divide=divide)
        return self
