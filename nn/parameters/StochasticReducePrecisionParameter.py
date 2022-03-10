import torch

from nn.fn.reduce_precision import stochastic_reduce_precision
from nn.parameters.PseudoParameter import PseudoParameter


class StochasticReducePrecisionParameter(PseudoParameter):
    precision: int

    def __init__(self, data=None, requires_grad=True, precision: int = 8, initialise_zero_pseudo=False):
        self.precision = precision
        super(StochasticReducePrecisionParameter, self).__init__(data, requires_grad, initialise_zero_pseudo)

    def __repr__(self):
        return f'StochasticReducePrecisionParameter(precision: {self.precision}):\n' + super(
            StochasticReducePrecisionParameter, self).__repr__()

    @torch.no_grad()
    def set_data(self, data, precision=None):
        if precision is None:
            precision = self.precision
        self.data = stochastic_reduce_precision(data, precision=precision)
        return self
