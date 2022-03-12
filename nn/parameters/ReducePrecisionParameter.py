from nn.layers.ReducePrecision import ReducePrecision
from nn.parameters.PseudoParameter import PseudoParameter


class ReducePrecisionParameter(PseudoParameter):
    precision: int
    divide: float

    def __init__(self, data=None, requires_grad=True, precision: int = 8, divide: float = 0.5,
                 initialise_zero_pseudo=False):
        self.precision = precision
        self.divide = divide
        super(ReducePrecisionParameter, self).__init__(
            data=data,
            requires_grad=requires_grad,
            transform=ReducePrecision(precision=precision, divide=divide),
            initialise_zero_pseudo=initialise_zero_pseudo
        )

    def __repr__(self):
        return f'ReducePrecisionParameter(precision: {self.precision}, divide:{self.divide}):\n' + super(
            ReducePrecisionParameter, self).__repr__()
