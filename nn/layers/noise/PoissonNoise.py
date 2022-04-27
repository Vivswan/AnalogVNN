from typing import Union

import numpy as np
import scipy.special
import torch
from torch import Tensor

from nn.backward_pass.BackwardFunction import BackwardIdentity
from nn.layers.BaseLayer import BaseLayer
from nn.utils.to_tensor_parameter import to_float_tensor, to_nongrad_parameter
from nn.utils.types import TENSOR_OPERABLE
from utils.dirac_delta import dirac_delta


class PoissonNoise(BaseLayer, BackwardIdentity):
    def __init__(
            self,
            scale: Union[None, int, float] = 1.,
            precision: Union[None, int] = None,
    ):
        super(PoissonNoise, self).__init__()

        self.scale, self.precision = to_float_tensor(
            scale, precision
        )
        self.scale, self.precision = to_nongrad_parameter(
            self.scale, self.precision
        )

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
                correctness += PoissonNoise.staticmethod_cdf(
                    max(abs(max_i), abs(min_i)), rate=i, scale_factor=scale * precision
                )
            else:
                correctness += PoissonNoise.staticmethod_cdf(
                    max(abs(max_i), abs(min_i)), rate=i, scale_factor=scale * precision
                )
                correctness -= PoissonNoise.staticmethod_cdf(
                    min(abs(max_i), abs(min_i)), rate=i, scale_factor=scale * precision
                )

        correctness /= cdf_domain.size - 1
        return 1 - correctness

    @property
    def leakage(self):
        return self.staticmethod_leakage(scale=float(self.scale), precision=int(self.precision))

    @property
    def scale_factor(self):
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

        x = torch.abs(x) * self.scale_factor
        rate = torch.abs(rate) * self.scale_factor
        return x.xlogy(rate) - rate - torch.lgamma(x + 1)

    @staticmethod
    def staticmethod_cdf(x: TENSOR_OPERABLE, rate: TENSOR_OPERABLE, scale_factor: TENSOR_OPERABLE):
        if np.isclose(rate, np.zeros_like(rate)):
            return np.ones_like(x)

        x = np.abs(x) * scale_factor
        rate = np.abs(rate) * scale_factor
        return scipy.special.gammaincc(x + 1, rate)

    def cdf(self, x: Tensor, rate: Tensor) -> Tensor:
        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        rate = rate if isinstance(rate, Tensor) else torch.tensor(rate, requires_grad=False)
        return self.staticmethod_cdf(x, rate, self.scale_factor)

    def forward(self, x: Tensor) -> Tensor:
        y = torch.sign(x)
        y *= torch.poisson(torch.abs(x * self.scale_factor))

        if self.scale_factor > 1.:
            y /= self.scale_factor
        else:
            y *= self.scale_factor

        return y

    def extra_repr(self) -> str:
        return f'scale={self.scale:.4f}, leakage={self.leakage}, precision={self.precision}'
