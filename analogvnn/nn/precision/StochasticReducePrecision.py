import torch
from torch import nn, Tensor

from analogvnn.backward.BackwardIdentity import BackwardIdentity
from analogvnn.fn.reduce_precision import stochastic_reduce_precision
from analogvnn.nn.precision.Precision import Precision
from analogvnn.utils.common_types import TENSOR_OPERABLE

__all__ = ['StochasticReducePrecision']


class StochasticReducePrecision(Precision, BackwardIdentity):
    """Implements the stochastic reduce precision function.

    Attributes:
        precision (nn.Parameter): the precision of the output tensor.
    """

    __constants__ = ['precision']
    precision: nn.Parameter

    def __init__(self, precision: int = 8):
        """Initialize the StochasticReducePrecision module.

        Args:
            precision (int): the precision of the output tensor.
        """

        super(StochasticReducePrecision, self).__init__()
        if precision < 1:
            raise ValueError('precision has to be more than 0, but got {}'.format(precision))

        if precision != int(precision):
            raise ValueError('precision must be int, but got {}'.format(precision))

        self.precision = nn.Parameter(torch.tensor(precision), requires_grad=False)

    @property
    def precision_width(self) -> Tensor:
        """The precision width.

        Returns:
            Tensor: the precision width
        """

        return 1 / self.precision

    @property
    def bit_precision(self) -> Tensor:
        """The bit precision of the ReducePrecision module.

        Returns:
            Tensor: the bit precision of the ReducePrecision module.
        """

        return torch.log2(self.precision + 1)

    @staticmethod
    def convert_to_precision(bit_precision: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
        """Convert the bit precision to the precision.

        Args:
            bit_precision (TENSOR_OPERABLE): the bit precision.

        Returns:
            TENSOR_OPERABLE: the precision.
        """

        return 2 ** bit_precision - 1

    def extra_repr(self) -> str:
        """The extra __repr__ string of the StochasticReducePrecision module.

        Returns:
            str: string
        """

        return f'precision={self.precision}'

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of the StochasticReducePrecision module.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: output tensor.
        """

        return stochastic_reduce_precision(x, self.precision)
