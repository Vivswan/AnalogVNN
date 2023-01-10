import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

from analogvnn.backward.BackwardIdentity import BackwardIdentity
from analogvnn.nn.module.Layer import Layer
from analogvnn.utils.common_types import TENSOR_OPERABLE
from analogvnn.utils.to_tensor_parameter import to_float_tensor, to_nongrad_parameter

__all__ = ['LaplacianNoise']


class LaplacianNoise(Layer, BackwardIdentity):
    """Implements the Laplacian noise function.

    Attributes:
        scale (nn.Parameter): the scale of the Laplacian noise.
        leakage (nn.Parameter): the leakage of the Laplacian noise.
        precision (nn.Parameter): the precision of the Laplacian noise.
    """
    __constants__ = ['scale', 'leakage', 'precision']
    scale: nn.Parameter
    leakage: nn.Parameter
    precision: nn.Parameter

    def __init__(
            self,
            scale: Optional[float] = None,
            leakage: Optional[float] = None,
            precision: Optional[int] = None
    ):
        """initialize the Laplacian noise function.

        Args:
            scale (float): the scale of the Laplacian noise.
            leakage (float): the leakage of the Laplacian noise.
            precision (int): the precision of the Laplacian noise.
        """
        super(LaplacianNoise, self).__init__()

        if (scale is None) + (leakage is None) + (precision is None) != 1:
            raise ValueError("only 2 out of 3 arguments are needed (scale, leakage, precision)")

        scale, leakage, precision = to_float_tensor(scale, leakage, precision)

        if scale is None and leakage is not None and precision is not None:
            scale = self.calc_scale(leakage, precision)

        if precision is None and scale is not None and leakage is not None:
            precision = self.calc_precision(scale, leakage)

        if leakage is None and scale is not None and precision is not None:
            leakage = self.calc_leakage(scale, precision)

        self.scale, self.leakage, self.precision = to_nongrad_parameter(scale, leakage, precision)

    @staticmethod
    def calc_scale(leakage: TENSOR_OPERABLE, precision: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
        """calculate the scale of the Laplacian noise.

        Args:
            leakage (float): the leakage of the Laplacian noise.
            precision (int): the precision of the Laplacian noise.

        Returns:
            float: the scale of the Laplacian noise.
        """
        return - 1 / (2 * math.log(leakage) * precision)

    @staticmethod
    def calc_precision(scale: TENSOR_OPERABLE, leakage: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
        """calculate the precision of the Laplacian noise.

        Args:
            scale (float): the scale of the Laplacian noise.
            leakage (float): the leakage of the Laplacian noise.

        Returns:
            int: the precision of the Laplacian noise.
        """
        return - 1 / (2 * math.log(leakage) * scale)

    @staticmethod
    def calc_leakage(scale: TENSOR_OPERABLE, precision: TENSOR_OPERABLE) -> Tensor:
        """calculate the leakage of the Laplacian noise.

        Args:
            scale (float): the scale of the Laplacian noise.
            precision (int): the precision of the Laplacian noise.

        Returns:
            float: the leakage of the Laplacian noise.
        """
        return 2 * LaplacianNoise.static_cdf(x=-1 / (2 * precision), scale=scale)

    @property
    def stddev(self) -> Tensor:
        """the standard deviation of the Laplacian noise.

        Returns:
            Tensor: the standard deviation of the Laplacian noise.
        """
        return (2 ** 0.5) * self.scale

    @property
    def variance(self) -> Tensor:
        """the variance of the Laplacian noise.

        Returns:
            Tensor: the variance of the Laplacian noise.
        """
        return 2 * self.scale.pow(2)

    def pdf(self, x: TENSOR_OPERABLE, loc: TENSOR_OPERABLE = 0) -> Tensor:
        """the probability density function of the Laplacian noise.

        Args:
            x (TENSOR_OPERABLE): the input tensor.
            loc (TENSOR_OPERABLE): the mean of the Laplacian noise.

        Returns:
            Tensor: the probability density function of the Laplacian noise.
        """
        return torch.exp(self.log_prob(x=x, loc=loc))

    def log_prob(self, x: TENSOR_OPERABLE, loc: TENSOR_OPERABLE = 0) -> Tensor:
        """the log probability density function of the Laplacian noise.

        Args:
            x (TENSOR_OPERABLE): the input tensor.
            loc (TENSOR_OPERABLE): the mean of the Laplacian noise.

        Returns:
            Tensor: the log probability density function of the Laplacian noise.
        """
        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        loc = loc if isinstance(loc, Tensor) else torch.tensor(loc, requires_grad=False)
        return -torch.log(2 * self.scale) - torch.abs(x - loc) / self.scale

    @staticmethod
    def static_cdf(x: TENSOR_OPERABLE, scale: TENSOR_OPERABLE, loc: TENSOR_OPERABLE = 0.) -> TENSOR_OPERABLE:
        """the cumulative distribution function of the Laplacian noise.

        Args:
            x (TENSOR_OPERABLE): the input tensor.
            scale (TENSOR_OPERABLE): the scale of the Laplacian noise.
            loc (TENSOR_OPERABLE): the mean of the Laplacian noise.

        Returns:
            TENSOR_OPERABLE: the cumulative distribution function of the Laplacian noise.
        """
        return 0.5 - 0.5 * np.sign(x - loc) * np.expm1(-abs(x - loc) / scale)

    def cdf(self, x: Tensor, loc: Tensor = 0) -> Tensor:
        """the cumulative distribution function of the Laplacian noise.

        Args:
            x (Tensor): the input tensor.
            loc (Tensor): the mean of the Laplacian noise.

        Returns:
            Tensor: the cumulative distribution function of the Laplacian noise.
        """
        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        loc = loc if isinstance(loc, Tensor) else torch.tensor(loc, requires_grad=False)
        return self.static_cdf(x=x, scale=self.scale, loc=loc)

    def forward(self, x: Tensor) -> Tensor:
        """add Laplacian noise to the input tensor.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor with Laplacian noise.
        """
        return torch.distributions.Laplace(loc=x, scale=self.scale).sample()

    def extra_repr(self) -> str:
        """the extra representation of the Laplacian noise.

        Returns:
            str: the extra representation of the Laplacian noise.
        """
        return f'std={float(self.std):.4f}, leakage={float(self.leakage):.4f}, precision={int(self.precision)}'
