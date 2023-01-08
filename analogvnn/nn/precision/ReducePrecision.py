import torch
from torch import nn, Tensor

from analogvnn.backward.BackwardIdentity import BackwardIdentity
from analogvnn.fn.reduce_precision import reduce_precision
from analogvnn.nn.module.Layer import Layer
from analogvnn.utils.common_types import TENSOR_OPERABLE

__all__ = ['ReducePrecision']


class ReducePrecision(Layer, BackwardIdentity):
    """Implements the reduce precision function.

    Attributes:
        precision (nn.Parameter): the precision of the output tensor.
        divide (nn.Parameter): the rounding value that is if divide is 0.5,
         then 0.6 will be rounded to 1.0 and 0.4 will be rounded to 0.0.
    """
    __constants__ = ['precision', 'divide']
    precision: nn.Parameter
    divide: nn.Parameter

    def __init__(self, precision: int = None, divide: float = 0.5):
        """initialize the reduce precision function.

        Args:
            precision (int): the precision of the output tensor.
            divide (float): the rounding value that is if divide is 0.5,
             then 0.6 will be rounded to 1.0 and 0.4 will be rounded to 0.0.
        """
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
    def precision_width(self) -> Tensor:
        """the precision width

        Returns:
            Tensor: the precision width
        """
        return 1 / self.precision

    @property
    def bit_precision(self) -> Tensor:
        """the bit precision of the ReducePrecision module.

        Returns:
            Tensor: the bit precision of the ReducePrecision module.
        """
        return torch.log2(self.precision + 1)

    @staticmethod
    def convert_to_precision(bit_precision: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
        """convert the bit precision to the precision.

        Args:
            bit_precision (TENSOR_OPERABLE): the bit precision.

        Returns:
            TENSOR_OPERABLE: the precision.
        """
        return 2 ** bit_precision - 1

    def extra_repr(self) -> str:
        """the extra __repr__ string of the ReducePrecision module.

        Returns:
            str: string
        """
        return f'precision={int(self.precision)}, divide={float(self.divide):0.2f}'

    def forward(self, x: Tensor) -> Tensor:
        """forward function of the ReducePrecision module.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """
        return reduce_precision(x, self.precision, self.divide)
