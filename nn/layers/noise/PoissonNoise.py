from typing import Union

import numpy as np
import scipy.special
import torch
from scipy.optimize import *
from torch import Tensor

from nn.fn.BackwardIdentity import BackwardIdentity
from nn.fn.dirac_delta import dirac_delta
from nn.modules.Layer import Layer
from nn.utils.common_types import TENSOR_OPERABLE
from nn.utils.to_tensor_parameter import to_float_tensor, to_nongrad_parameter


class PoissonNoise(Layer, BackwardIdentity):
    def __init__(
            self,
            scale: Union[None, float] = None,
            max_leakage: Union[None, float] = None,
            precision: Union[None, int] = None,
    ):
        super(PoissonNoise, self).__init__()

        if (scale is None) + (max_leakage is None) + (precision is None) != 1:
            raise ValueError("only 2 out of 3 arguments are needed (scale, max_leakage, precision)")

        self.scale, self.max_leakage, self.precision = to_float_tensor(
            scale, max_leakage, precision
        )

        if self.scale is None and self.max_leakage is not None and self.precision is not None:
            self.scale = self.calc_scale(self.max_leakage, self.precision)

        if self.precision is None and self.scale is not None and self.max_leakage is not None:
            self.precision = self.calc_precision(self.scale, self.max_leakage)

        if self.max_leakage is None and self.scale is not None and self.precision is not None:
            self.max_leakage = self.calc_max_leakage(self.scale, self.precision)

        self.scale, self.max_leakage, self.precision = to_nongrad_parameter(
            self.scale, self.max_leakage, self.precision
        )

        if self.rate_factor < 1.:
            raise ValueError("rate_factor must be >= 1 (scale * precision >= 1)")

    @staticmethod
    def calc_scale(max_leakage, precision, max_check=10000):
        max_leakage = float(max_leakage)
        precision = float(precision)
        return toms748(
            lambda s: PoissonNoise.calc_max_leakage(s, precision) - max_leakage,
            a=0,
            b=max_check,
            maxiter=10000
        )

    @staticmethod
    def calc_precision(scale, max_leakage, max_check=2 ** 16):
        max_leakage = float(max_leakage)
        scale = float(scale)
        return toms748(
            lambda p: PoissonNoise.calc_max_leakage(scale, p) - max_leakage,
            a=1,
            b=max_check,
            maxiter=10000
        )

    @staticmethod
    def calc_max_leakage(scale, precision):
        return 1 - (
                PoissonNoise.static_cdf(x=1. + 1 / (2 * precision), rate=1., scale_factor=scale * precision)
                - PoissonNoise.static_cdf(x=1. - 1 / (2 * precision), rate=1., scale_factor=scale * precision)
        )

    @staticmethod
    def static_cdf(x: TENSOR_OPERABLE, rate: TENSOR_OPERABLE, scale_factor: TENSOR_OPERABLE):
        if np.isclose(rate, np.zeros_like(rate)):
            return np.ones_like(x)

        x = np.abs(x) * scale_factor
        rate = np.abs(rate) * scale_factor
        return scipy.special.gammaincc(x + 1, rate)

    @staticmethod
    def staticmethod_leakage(scale, precision):
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
    def leakage(self):
        return self.staticmethod_leakage(scale=float(self.scale), precision=int(self.precision))

    @property
    def rate_factor(self):
        return self.precision * self.scale

    def pdf(self, x: Tensor, rate: Tensor) -> Tensor:
        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        rate = rate if isinstance(rate, Tensor) else torch.tensor(rate, requires_grad=False)

        if torch.isclose(rate, torch.zeros_like(rate)):
            return dirac_delta(x)

        return torch.exp(self.log_prob(x=x, rate=rate))

    def log_prob(self, x: Tensor, rate: Tensor) -> Tensor:
        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        rate = rate if isinstance(rate, Tensor) else torch.tensor(rate, requires_grad=False)

        x = torch.abs(x) * self.rate_factor
        rate = torch.abs(rate) * self.rate_factor
        return x.xlogy(rate) - rate - torch.lgamma(x + 1)

    def cdf(self, x: Tensor, rate: Tensor) -> Tensor:
        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        rate = rate if isinstance(rate, Tensor) else torch.tensor(rate, requires_grad=False)
        return self.static_cdf(x, rate, self.rate_factor)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sign(x) * torch.poisson(torch.abs(x * self.rate_factor)) / self.rate_factor

    def extra_repr(self) -> str:
        return f'scale={float(self.scale):.4f}' \
               f', max_leakage={float(self.max_leakage):.4f}' \
               f', leakage={float(self.leakage):.4f}' \
               f', precision={int(self.precision)}'
