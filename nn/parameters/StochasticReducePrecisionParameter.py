import torch

from nn.layers.StochasticReducePrecision import stochastic_reduce_precision
from nn.parameters.BasePrecisionParameter import BasePrecisionParameter


class StochasticReducePrecisionParameter(BasePrecisionParameter):
    precision: int

    def __init__(self, data=None, requires_grad=True, precision: int = 8, use_zero_pseudo_tensor=False):
        self.precision = precision
        super(StochasticReducePrecisionParameter, self).__init__(data, requires_grad, use_zero_pseudo_tensor)

    def __repr__(self):
        return f'StochasticReducePrecisionParameter(precision: {self.precision}): ' + super(
            StochasticReducePrecisionParameter, self).__repr__()

    @torch.no_grad()
    def set_tensor(self, data, precision=None):
        if precision is None:
            precision = self.precision
        self.data = stochastic_reduce_precision(data, precision=precision)
        return self
