import math
from typing import Union

import numpy as np
import torch
from torch import Tensor

from nn.layers.BackwardIdentity import BackwardIdentity
from nn.modules.Layer import Layer
from nn.utils.common_types import TENSOR_OPERABLE
from nn.utils.to_tensor_parameter import to_float_tensor, to_nongrad_parameter


class LaplacianNoise(Layer, BackwardIdentity):
    __constants__ = ['scale', 'leakage', 'precision']

    def __init__(
            self,
            scale: Union[None, float] = None,
            leakage: Union[None, float] = None,
            precision: Union[None, int] = None
    ):
        super(LaplacianNoise, self).__init__()

        if scale is not None and precision is not None and leakage is not None:
            raise ValueError("only 2 out of 3 arguments are needed (scale, precision, leakage)")

        if scale is None or (precision is None and leakage is None):
            raise ValueError("Invalid arguments not found: scale or (leakage and precision)")

        self.scale, self.leakage, self.precision = to_float_tensor(
            scale, leakage, precision
        )

        if self.scale is None and self.leakage is not None and self.precision is not None:
            self.scale = self.calc_scale(self.leakage, self.precision)

        if self.precision is None and self.scale is not None and self.leakage is not None:
            self.precision = self.calc_precision(self.scale, self.leakage)

        if self.leakage is None and self.scale is not None and self.precision is not None:
            self.leakage = self.calc_leakage(self.scale, self.precision)

        self.scale, self.leakage, self.precision = to_nongrad_parameter(
            self.scale, self.leakage, self.precision
        )

    @staticmethod
    def calc_scale(leakage, precision):
        return - 1 / (2 * math.log(leakage) * precision)

    @staticmethod
    def calc_precision(scale, leakage):
        return - 1 / (2 * math.log(leakage) * scale)

    @staticmethod
    def calc_leakage(scale, precision):
        # return math.exp((-1 / (2 * precision)) * (1 / scale))
        return 2 * LaplacianNoise.static_cdf(x=-1 / (2 * precision), scale=scale)

    @property
    def variance(self):
        return 2 * self.scale.pow(2)

    @property
    def stddev(self):
        return (2 ** 0.5) * self.scale

    def pdf(self, x: Tensor, loc: Tensor = 0) -> Tensor:
        return torch.exp(self.log_prob(x=x, loc=loc))

    def log_prob(self, x: Tensor, loc: Tensor = 0):
        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        loc = loc if isinstance(loc, Tensor) else torch.tensor(loc, requires_grad=False)
        return -torch.log(2 * self.scale) - torch.abs(x - loc) / self.scale

    @staticmethod
    def static_cdf(x: TENSOR_OPERABLE, scale: TENSOR_OPERABLE, loc: TENSOR_OPERABLE = 0.):
        return 0.5 - 0.5 * np.sign(x - loc) * np.expm1(-abs(x - loc) / scale)

    def cdf(self, x: Tensor, loc: Tensor = 0) -> Tensor:
        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        loc = loc if isinstance(loc, Tensor) else torch.tensor(loc, requires_grad=False)
        return self.static_cdf(x=x, scale=self.scale, loc=loc)

    def forward(self, x: Tensor) -> Tensor:
        return torch.distributions.Laplace(loc=x, scale=self.scale).sample()

    def extra_repr(self) -> str:
        return f'std={float(self.std):.4f}, leakage={float(self.leakage):.4f}, precision={int(self.precision)}'
