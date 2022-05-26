from nn.layers.functionals.StochasticReducePrecision import StochasticReducePrecision
from nn.parameters.PseudoParameter import PseudoParameter


class StochasticReducePrecisionParameter(PseudoParameter):
    precision: int

    def __init__(self, data=None, requires_grad=True, precision: int = 8, initialise_zero_pseudo=False):
        self.precision = precision
        super(StochasticReducePrecisionParameter, self).__init__(
            data=data,
            requires_grad=requires_grad,
            transform=StochasticReducePrecision(precision=precision),
            initialise_zero_pseudo=initialise_zero_pseudo
        )

    def __repr__(self):
        return f'StochasticReducePrecisionParameter(precision: {self.precision}):\n' + super(
            StochasticReducePrecisionParameter, self).__repr__()
