import math
from numbers import Real
from typing import Optional

import scipy.special
import torch
from torch import Tensor, nn

from analogvnn.backward.BackwardIdentity import BackwardIdentity
from analogvnn.nn.noise.Noise import Noise
from analogvnn.utils.common_types import TENSOR_OPERABLE
from analogvnn.utils.to_tensor_parameter import to_nongrad_parameter, to_float_tensor

__all__ = ['GaussianNoise']


class GaussianNoise(Noise, BackwardIdentity):
    """Implements the Gaussian noise function.

    Attributes:
        std (nn.Parameter): the standard deviation of the Gaussian noise.
        leakage (nn.Parameter): the leakage of the Gaussian noise.
        precision (nn.Parameter): the precision of the Gaussian noise.
    """

    __constants__ = ['std', 'leakage', 'precision']
    std: nn.Parameter
    leakage: nn.Parameter
    precision: nn.Parameter

    def __init__(
            self,
            std: Optional[float] = None,
            leakage: Optional[float] = None,
            precision: Optional[int] = None
    ):
        """Initialize the Gaussian noise function.

        Args:
            std (float): the standard deviation of the Gaussian noise.
            leakage (float): the leakage of the Gaussian noise.
            precision (int): the precision of the Gaussian noise.
        """

        super().__init__()

        if (std is None) + (leakage is None) + (precision is None) != 1:
            raise ValueError('only 2 out of 3 arguments are needed (std, leakage, precision)')

        std, leakage, precision = to_float_tensor(std, leakage, precision)

        if std is None and leakage is not None and precision is not None:
            std = self.calc_std(leakage, precision)

        if precision is None and std is not None and leakage is not None:
            precision = self.calc_precision(std, leakage)

        if leakage is None and std is not None and precision is not None:
            leakage = self.calc_leakage(std, precision)

        self.std, self.leakage, self.precision = to_nongrad_parameter(std, leakage, precision)

    @staticmethod
    def calc_std(leakage: TENSOR_OPERABLE, precision: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
        """Calculate the standard deviation of the Gaussian noise.

        Args:
            leakage (float): the leakage of the Gaussian noise.
            precision (int): the precision of the Gaussian noise.

        Returns:
            float: the standard deviation of the Gaussian noise.
        """

        return 1 / (2 * precision * scipy.special.erfinv(1 - leakage) * math.sqrt(2))

    @staticmethod
    def calc_precision(std: TENSOR_OPERABLE, leakage: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
        """Calculate the precision of the Gaussian noise.

        Args:
            std (float): the standard deviation of the Gaussian noise.
            leakage (float): the leakage of the Gaussian noise.

        Returns:
            int: the precision of the Gaussian noise.
        """

        return 1 / (2 * std * scipy.special.erfinv(1 - leakage) * math.sqrt(2))

    @staticmethod
    def calc_leakage(std: TENSOR_OPERABLE, precision: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
        """Calculate the leakage of the Gaussian noise.

        Args:
            std (float): the standard deviation of the Gaussian noise.
            precision (int): the precision of the Gaussian noise.

        Returns:
            float: the leakage of the Gaussian noise.
        """

        return 2 * GaussianNoise.static_cdf(x=-1 / (2 * precision), std=std)

    @property
    def stddev(self) -> Tensor:
        """The standard deviation of the Gaussian noise.

        Returns:
            Tensor: the standard deviation of the Gaussian noise.
        """

        return self.std

    @property
    def variance(self) -> Tensor:
        """The variance of the Gaussian noise.

        Returns:
            Tensor: the variance of the Gaussian noise.
        """

        return self.stddev.pow(2)

    def pdf(self, x: Tensor, mean: Tensor = 0) -> Tensor:
        """Calculate the probability density function of the Gaussian noise.

        Args:
            x (Tensor): the input tensor.
            mean (Tensor): the mean of the Gaussian noise.

        Returns:
            Tensor: the probability density function of the Gaussian noise.
        """

        return torch.exp(self.log_prob(x=x, mean=mean))

    def log_prob(self, x: Tensor, mean: Tensor = 0) -> Tensor:
        """Calculate the log probability density function of the Gaussian noise.

        Args:
            x (Tensor): the input tensor.
            mean (Tensor): the mean of the Gaussian noise.

        Returns:
            Tensor: the log probability density function of the Gaussian noise.
        """

        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        mean = mean if isinstance(mean, Tensor) else torch.tensor(mean, requires_grad=False)

        var = (self.std ** 2)
        log_scale = math.log(self.std) if isinstance(self.std, Real) else self.std.log()
        return -((x - mean) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    @staticmethod
    def static_cdf(x: TENSOR_OPERABLE, std: TENSOR_OPERABLE, mean: TENSOR_OPERABLE = 0.) -> TENSOR_OPERABLE:
        """Calculate the cumulative distribution function of the Gaussian noise.

        Args:
            x (TENSOR_OPERABLE): the input tensor.
            std (TENSOR_OPERABLE): the standard deviation of the Gaussian noise.
            mean (TENSOR_OPERABLE): the mean of the Gaussian noise.

        Returns:
            TENSOR_OPERABLE: the cumulative distribution function of the Gaussian noise.
        """

        return 1 / 2 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))

    def cdf(self, x: Tensor, mean: Tensor = 0) -> Tensor:
        """Calculate the cumulative distribution function of the Gaussian noise.

        Args:
            x (Tensor): the input tensor.
            mean (Tensor): the mean of the Gaussian noise.

        Returns:
            Tensor: the cumulative distribution function of the Gaussian noise.
        """

        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        mean = mean if isinstance(mean, Tensor) else torch.tensor(mean, requires_grad=False)
        return self.static_cdf(x=x, std=self.std, mean=mean)

    def forward(self, x: Tensor) -> Tensor:
        """Add the Gaussian noise to the input tensor.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """

        return torch.normal(mean=x, std=self.std)

    def extra_repr(self) -> str:
        """The extra representation of the Gaussian noise.

        Returns:
            str: the extra representation of the Gaussian noise.
        """

        return f'std={float(self.std):.4f}, leakage={float(self.leakage):.4f}, precision={int(self.precision)}'
