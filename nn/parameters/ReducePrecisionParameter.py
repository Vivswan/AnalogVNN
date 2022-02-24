import torch

from nn.fn.reduce_precision import reduce_precision
from nn.parameters.PseudoParameter import PseudoParameter


class ReducePrecisionParameter(PseudoParameter):
    precision: int
    divide: float

    def __init__(self, data=None, requires_grad=True, precision: int = 8, divide: float = 0.5,
                 initialise_zero_pseudo=False):
        self.precision = precision
        self.divide = divide
        super(ReducePrecisionParameter, self).__init__(data, requires_grad, initialise_zero_pseudo)

    def __repr__(self):
        return f'ReducePrecisionParameter(precision: {self.precision}, divide:{self.divide}):\n' + super(
            ReducePrecisionParameter, self).__repr__()

    @torch.no_grad()
    def set_data(self, data=None, precision=None, divide=None):
        if precision is None:
            precision = self.precision
        if divide is None:
            divide = self.divide
        self.data = reduce_precision(data, precision=precision, divide=divide)
        return self


if __name__ == '__main__':
    p = ReducePrecisionParameter(data=torch.eye(2), precision=4, divide=0.1)
    print(p)
