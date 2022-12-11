import math
from numbers import Real
from typing import Union

import scipy.special
import torch
from torch import Tensor

from nn.fn.BackwardIdentity import BackwardIdentity
from nn.modules.Layer import Layer
from nn.utils.common_types import TENSOR_OPERABLE
from nn.utils.to_tensor_parameter import to_nongrad_parameter, to_float_tensor


class GaussianNoise(Layer, BackwardIdentity):
    __constants__ = ['std', 'leakage', 'precision']

    def __init__(
            self,
            std: Union[None, int, float] = None,
            leakage: Union[None, int, float] = None,
            precision: Union[None, int] = None
    ):
        super(GaussianNoise, self).__init__()

        if std is not None and precision is not None and leakage is not None:
            raise ValueError("only 2 out of 3 arguments are needed (std, precision, leakage)")

        if std is None and precision is None and leakage is None:
            raise ValueError("only 2 out of 3 arguments are needed (std, precision, leakage)")

        if std is None and precision is None:
            raise ValueError("only 2 out of 3 arguments are needed (std, precision, leakage)")

        if precision is None and leakage is None:
            raise ValueError("only 2 out of 3 arguments are needed (std, precision, leakage)")

        if leakage is None and std is None:
            raise ValueError("only 2 out of 3 arguments are needed (std, precision, leakage)")

        self.std, self.leakage, self.precision = to_float_tensor(
            std, leakage, precision
        )

        if self.std is None and self.leakage is not None and self.precision is not None:
            self.std = self.calc_std(self.leakage, self.precision)

        if self.precision is None and self.std is not None and self.leakage is not None:
            self.precision = self.calc_precision(self.std, self.leakage)

        if self.leakage is None and self.std is not None and self.precision is not None:
            self.leakage = self.calc_leakage(self.std, self.precision)

        self.std, self.leakage, self.precision = to_nongrad_parameter(
            self.std, self.leakage, self.precision
        )

    @staticmethod
    def calc_std(leakage, precision):
        return 1 / (2 * precision * scipy.special.erfinv(1 - leakage) * math.sqrt(2))

    @staticmethod
    def calc_precision(std, leakage):
        return 1 / (2 * std * scipy.special.erfinv(1 - leakage) * math.sqrt(2))

    @staticmethod
    def calc_leakage(std, precision):
        return 2 * GaussianNoise.static_cdf(x=-1 / (2 * precision), std=std)

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def pdf(self, x: Tensor, loc: Tensor = 0) -> Tensor:
        return torch.exp(self.log_prob(x=x, mean=loc))

    def log_prob(self, x: Tensor, mean: Tensor = 0) -> Tensor:
        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        mean = mean if isinstance(mean, Tensor) else torch.tensor(mean, requires_grad=False)

        var = (self.std ** 2)
        log_scale = math.log(self.std) if isinstance(self.std, Real) else self.std.log()
        return -((x - mean) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    @staticmethod
    def static_cdf(x: TENSOR_OPERABLE, std: TENSOR_OPERABLE, mean: TENSOR_OPERABLE = 0.):
        return 1 / 2 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))

    def cdf(self, x: Tensor, mean: Tensor = 0) -> Union[Tensor, float]:
        x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
        mean = mean if isinstance(mean, Tensor) else torch.tensor(mean, requires_grad=False)
        return self.static_cdf(x=x, std=self.std, mean=mean)

    def forward(self, x: Tensor) -> Tensor:
        return torch.normal(mean=x, std=self.std)

    def extra_repr(self) -> str:
        return f'std={float(self.std):.4f}, leakage={float(self.leakage):.4f}, precision={int(self.precision)}'
