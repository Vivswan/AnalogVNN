from typing import Optional

import numpy as np
import scipy.special
import torch
from scipy.optimize import toms748
from torch import Tensor, nn

from analogvnn.backward.BackwardIdentity import BackwardIdentity
from analogvnn.fn.dirac_delta import gaussian_dirac_delta
from analogvnn.nn.noise.Noise import Noise
from analogvnn.utils.common_types import TENSOR_OPERABLE
from analogvnn.utils.to_tensor_parameter import to_float_tensor, to_nongrad_parameter

__all__ = ['PoissonNoise']


class PoissonNoise(Noise, BackwardIdentity):
    """Implements the Poisson noise function.

    Attributes:
        scale (nn.Parameter): the scale of the Poisson noise function.
        max_leakage (nn.Parameter): the maximum leakage of the Poisson noise.
        precision (nn.Parameter): the precision of the Poisson noise.
    """

    __constants__ = ['scale', 'max_leakage', 'precision']
    scale: nn.Parameter
    max_leakage: nn.Parameter
    precision: nn.Parameter

    def __init__(
            self,
            scale: Optional[float] = None,
            max_leakage: Optional[float] = None,
            precision: Optional[int] = None,
    ):
        """Initializes the Poisson noise function.

        Args:
            scale (Optional[float]): the scale of the Poisson noise function.
            max_leakage (Optional[float]): the maximum leakage of the Poisson noise.
            precision (Optional[int]): the precision of the Poisson noise.
        """

        super().__init__()

        if (scale is None) + (max_leakage is None) + (precision is None) != 1:
            raise ValueError('only 2 out of 3 arguments are needed (scale, max_leakage, precision)')

        scale, max_leakage, precision = to_float_tensor(scale, max_leakage, precision)

        if scale is None and max_leakage is not None and precision is not None:
            scale = self.calc_scale(max_leakage, precision)

        if precision is None and scale is not None and max_leakage is not None:
            precision = self.calc_precision(scale, max_leakage)

        if max_leakage is None and scale is not None and precision is not None:
            max_leakage = self.calc_max_leakage(scale, precision)

        self.scale, self.max_leakage, self.precision = to_nongrad_parameter(scale, max_leakage, precision)

        if self.rate_factor < 1.:
            raise ValueError('rate_factor must be >= 1 (scale * precision >= 1)')

    @staticmethod
    def calc_scale(max_leakage: TENSOR_OPERABLE, precision: TENSOR_OPERABLE, max_check=10000) -> TENSOR_OPERABLE:
        """Calculates the scale using the maximum leakage and the precision.

        Args:
            max_leakage (TENSOR_OPERABLE): the maximum leakage of the Poisson noise.
            precision (TENSOR_OPERABLE): the precision of the Poisson noise.
            max_check (int): the maximum value to check for the scale.

        Returns:
            TENSOR_OPERABLE: the scale of the Poisson noise function.
        """

        max_leakage = float(max_leakage)
        precision = float(precision)
        r, _ = toms748(
            lambda s: PoissonNoise.calc_max_leakage(s, precision) - max_leakage,
            a=0,
            b=max_check,
            maxiter=10000
        )
        return r

    @staticmethod
    def calc_precision(scale: TENSOR_OPERABLE, max_leakage: TENSOR_OPERABLE, max_check=2 ** 16) -> TENSOR_OPERABLE:
        """Calculates the precision using the scale and the maximum leakage.

        Args:
            scale (TENSOR_OPERABLE): the scale of the Poisson noise function.
            max_leakage (TENSOR_OPERABLE): the maximum leakage of the Poisson noise.
            max_check (int): the maximum value to check for the precision.

        Returns:
            TENSOR_OPERABLE: the precision of the Poisson noise.
        """

        max_leakage = float(max_leakage)
        scale = float(scale)
        r, _ = toms748(
            lambda p: PoissonNoise.calc_max_leakage(scale, p) - max_leakage,
            a=1,
            b=max_check,
            maxiter=10000
        )
        return r

    @staticmethod
    def calc_max_leakage(scale: TENSOR_OPERABLE, precision: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
        """Calculates the maximum leakage using the scale and the precision.

        Args:
            scale (TENSOR_OPERABLE): the scale of the Poisson noise function.
            precision (TENSOR_OPERABLE): the precision of the Poisson noise.

        Returns:
            TENSOR_OPERABLE: the maximum leakage of the Poisson noise.
        """

        return 1 - (
                PoissonNoise.static_cdf(x=1. + 1 / (2 * precision), rate=1., scale_factor=scale * precision)
                - PoissonNoise.static_cdf(x=1. - 1 / (2 * precision), rate=1., scale_factor=scale * precision)
        )

    @staticmethod
    def static_cdf(x: TENSOR_OPERABLE, rate: TENSOR_OPERABLE, scale_factor: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
        """Calculates the cumulative distribution function of the Poisson noise.

        Args:
            x (TENSOR_OPERABLE): the input of the Poisson noise.
            rate (TENSOR_OPERABLE): the rate of the Poisson noise.
            scale_factor (TENSOR_OPERABLE): the scale factor of the Poisson noise.

        Returns:
            TENSOR_OPERABLE: the cumulative distribution function of the Poisson noise.
        """

        if np.isclose(rate, np.zeros_like(rate)):
            return np.ones_like(x)

        x = np.abs(x) * scale_factor
        rate = np.abs(rate) * scale_factor
        return scipy.special.gammaincc(x + 1, rate)

    @staticmethod
    def staticmethod_leakage(scale: TENSOR_OPERABLE, precision: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
        """Calculates the leakage of the Poisson noise using the scale and the precision.

        Args:
            scale (TENSOR_OPERABLE): the scale of the Poisson noise function.
            precision (TENSOR_OPERABLE): the precision of the Poisson noise.

        Returns:
            TENSOR_OPERABLE: the leakage of the Poisson noise.
        """

        cdf_domain = np.linspace(-1, 1, int(2 * precision + 1), dtype=float)
        correctness = 0

        for i in cdf_domain:
            min_i = i - 1 / (2 * precision)
            max_i = i + 1 / (2 * precision)

            if np.isclose(i, 1.):
                max_i = 1.
            if np.isclose(i, -1.):
                min_i = -1.

            if np.isclose(i, 0.):
                correctness += 1
            else:
                correctness += PoissonNoise.static_cdf(
                    max(abs(max_i), abs(min_i)), rate=i, scale_factor=scale * precision
                )
                correctness -= PoissonNoise.static_cdf(
                    min(abs(max_i), abs(min_i)), rate=i, scale_factor=scale * precision
                )

        correctness /= cdf_domain.size - 1
        return 1 - correctness

    @property
    def leakage(self) -> float:
        """The leakage of the Poisson noise.

        Returns:
            float: the leakage of the Poisson noise.
        """

        return self.staticmethod_leakage(scale=float(self.scale), precision=int(self.precision))

    @property
    def rate_factor(self) -> Tensor:
        """The rate factor of the Poisson noise.

        Returns:
            Tensor: the rate factor of the Poisson noise.
        """

        return self.precision * self.scale

    def pdf(self, x: Tensor, rate: Tensor) -> Tensor:
        """Calculates the probability density function of the Poisson noise.

        Args:
            x (Tensor): the input of the Poisson noise.
            rate (Tensor): the rate of the Poisson noise.

        Returns:
            Tensor: the probability density function of the Poisson noise.
        """

        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        rate = rate if isinstance(rate, Tensor) else torch.tensor(rate, requires_grad=False)

        if torch.isclose(rate, torch.zeros_like(rate)):
            return gaussian_dirac_delta(x)

        return torch.exp(self.log_prob(x=x, rate=rate))

    def log_prob(self, x: Tensor, rate: Tensor) -> Tensor:
        """Calculates the log probability of the Poisson noise.

        Args:
            x (Tensor): the input of the Poisson noise.
            rate (Tensor): the rate of the Poisson noise.

        Returns:
            Tensor: the log probability of the Poisson noise.
        """

        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        rate = rate if isinstance(rate, Tensor) else torch.tensor(rate, requires_grad=False)

        x = torch.abs(x) * self.rate_factor
        rate = torch.abs(rate) * self.rate_factor
        return x.xlogy(rate) - rate - torch.lgamma(x + 1)

    def cdf(self, x: Tensor, rate: Tensor) -> Tensor:
        """Calculates the cumulative distribution function of the Poisson noise.

        Args:
            x (Tensor): the input of the Poisson noise.
            rate (Tensor): the rate of the Poisson noise.

        Returns:
            Tensor: the cumulative distribution function of the Poisson noise.
        """

        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        rate = rate if isinstance(rate, Tensor) else torch.tensor(rate, requires_grad=False)
        return self.static_cdf(x, rate, self.rate_factor)

    def forward(self, x: Tensor) -> Tensor:
        """Adds the Poisson noise to the input.

        Args:
            x (Tensor): the input of the Poisson noise.

        Returns:
            Tensor: the output of the Poisson noise.
        """

        return torch.sign(x) * torch.poisson(torch.abs(x * self.rate_factor)) / self.rate_factor

    def extra_repr(self) -> str:
        """Returns the extra representation of the Poisson noise.

        Returns:
            str: the extra representation of the Poisson noise.
        """

        return f'scale={float(self.scale):.4f}' \
               f', max_leakage={float(self.max_leakage):.4f}' \
               f', leakage={float(self.leakage):.4f}' \
               f', precision={int(self.precision)}'
