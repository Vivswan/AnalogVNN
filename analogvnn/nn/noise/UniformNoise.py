from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from analogvnn.backward.BackwardIdentity import BackwardIdentity
from analogvnn.nn.module.Layer import Layer
from analogvnn.utils.common_types import TENSOR_OPERABLE
from analogvnn.utils.to_tensor_parameter import to_float_tensor, to_nongrad_parameter

__all__ = ['UniformNoise']


class UniformNoise(Layer, BackwardIdentity):
    """Implements the uniform noise function.

    Attributes:
        low (nn.Parameter): the lower bound of the uniform noise.
        high (nn.Parameter): the upper bound of the uniform noise.
        leakage (nn.Parameter): the leakage of the uniform noise.
        precision (nn.Parameter): the precision of the uniform noise.
    """

    __constants__ = ['low', 'high', 'leakage', 'precision']
    low: nn.Parameter
    high: nn.Parameter
    leakage: nn.Parameter
    precision: nn.Parameter

    def __init__(
            self,
            low: Optional[float] = None,
            high: Optional[float] = None,
            leakage: Optional[float] = None,
            precision: Optional[int] = None
    ):
        """Initialize the uniform noise function.

        Args:
            low (float): the lower bound of the uniform noise.
            high (float): the upper bound of the uniform noise.
            leakage (float): the leakage of the uniform noise.
            precision (int): the precision of the uniform noise.
        """
        super(UniformNoise, self).__init__()

        if (low is None or high is None) + (leakage is None) + (precision is None) != 1:
            raise ValueError('only 2 out of 3 arguments are needed (scale, leakage, precision)')

        low, high, leakage, precision = to_float_tensor(low, high, leakage, precision)

        if (low is None or high is None) and leakage is not None and precision is not None:
            low, high = self.calc_high_low(leakage, precision)

        if low is not None and high is not None and leakage is None and precision is not None:
            leakage = self.calc_leakage(low, high, precision)

        if low is not None and high is not None and leakage is not None and precision is None:
            precision = self.calc_precision(low, high, leakage)

        self.low, self.high, self.leakage, self.precision = to_nongrad_parameter(low, high, leakage, precision)

    @staticmethod
    def calc_high_low(leakage: TENSOR_OPERABLE, precision: TENSOR_OPERABLE) -> Tuple[TENSOR_OPERABLE, TENSOR_OPERABLE]:
        """Calculate the high and low from leakage and precision.

        Args:
            leakage (TENSOR_OPERABLE): the leakage of the uniform noise.
            precision (TENSOR_OPERABLE): the precision of the uniform noise.

        Returns:
            Tuple[TENSOR_OPERABLE, TENSOR_OPERABLE]: the high and low of the uniform noise.
        """
        v = 1 / (1 - leakage) * (1 / precision)
        return -v / 2, v / 2

    @staticmethod
    def calc_leakage(low: TENSOR_OPERABLE, high: TENSOR_OPERABLE, precision: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
        """Calculate the leakage from low, high and precision.

        Args:
            low (TENSOR_OPERABLE): the lower bound of the uniform noise.
            high (TENSOR_OPERABLE): the upper bound of the uniform noise.
            precision (TENSOR_OPERABLE): the precision of the uniform noise.

        Returns:
            TENSOR_OPERABLE: the leakage of the uniform noise.
        """
        return 1 - min(1, (1 / precision) * (1 / (high - low)))

    @staticmethod
    def calc_precision(low: TENSOR_OPERABLE, high: TENSOR_OPERABLE, leakage: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
        """Calculate the precision from low, high and leakage.

        Args:
            low (TENSOR_OPERABLE): the lower bound of the uniform noise.
            high (TENSOR_OPERABLE): the upper bound of the uniform noise.
            leakage (TENSOR_OPERABLE): the leakage of the uniform noise.

        Returns:
            TENSOR_OPERABLE: the precision of the uniform noise.
        """
        return 1 / (1 - leakage) * (1 / (high - low))

    @property
    def mean(self) -> Tensor:
        """The mean of the uniform noise.

        Returns:
            Tensor: the mean of the uniform noise.
        """
        return (self.high + self.low) / 2

    @property
    def stddev(self) -> Tensor:
        """The standard deviation of the uniform noise.

        Returns:
            Tensor: the standard deviation of the uniform noise.
        """
        return (self.high - self.low) / 12 ** 0.5

    @property
    def variance(self) -> Tensor:
        """The variance of the uniform noise.

        Returns:
            Tensor: the variance of the uniform noise.
        """
        return (self.high - self.low).pow(2) / 12

    def pdf(self, x: Tensor) -> Tensor:
        """The probability density function of the uniform noise.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the probability density function of the uniform noise.
        """
        return torch.exp(self.log_prob(x=x))

    def log_prob(self, x: Tensor) -> Tensor:
        """The log probability density function of the uniform noise.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the log probability density function of the uniform noise.
        """
        lb = self.low.le(x).type_as(self.low)
        ub = self.high.gt(x).type_as(self.low)
        return torch.log(lb.mul(ub)) - torch.log(self.high - self.low)

    def cdf(self, x: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
        """The cumulative distribution function of the uniform noise.

        Args:
            x (TENSOR_OPERABLE): the input tensor.

        Returns:
            TENSOR_OPERABLE: the cumulative distribution function of the uniform noise.
        """
        result = (x - self.low) / (self.high - self.low)
        return result.clamp(min=0, max=1)

    def forward(self, x: Tensor) -> Tensor:
        """Add the uniform noise to the input tensor.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """
        return torch.distributions.Uniform(low=x + self.low, high=x + self.high).sample()

    def extra_repr(self) -> str:
        """The extra representation of the uniform noise.

        Returns:
            str: the extra representation of the uniform noise.
        """
        return f'high={float(self.high):.4f}' \
               f', low={float(self.low):.4f}' \
               f', leakage={float(self.leakage):.4f}' \
               f', precision={int(self.precision)}'
