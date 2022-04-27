import math
from numbers import Real
from typing import Union

import torch
from torch import Tensor

from nn.backward_pass.BackwardFunction import BackwardIdentity
from nn.layers.BaseLayer import BaseLayer
from nn.utils.to_tensor_parameter import to_nongrad_parameter, to_float_tensor
from nn.utils.types import TENSOR_OPERABLE


class GaussianNoise(BaseLayer, BackwardIdentity):
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

        if std is None or (precision is None and leakage is None):
            raise ValueError("Invalid arguments not found: std or (leakage and precision)")

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
        return 1 / (2 * precision * torch.erfinv(1 - leakage) * math.sqrt(2))

    @staticmethod
    def calc_precision(std, leakage):
        return 1 / (2 * std * torch.erfinv(1 - leakage) * math.sqrt(2))

    @staticmethod
    def calc_leakage(std, precision):
        # return 1 - torch.erf(1 / (std * 2 * precision * math.sqrt(2)))
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
        return f'std={self.std:.4f}, leakage={self.leakage}, precision={self.precision}'

    # def signal_to_noise_ratio(self, reference_std=1):
    #     return 1 / (reference_std * 2 * self.std)


if __name__ == '__main__':
    g = GaussianNoise(std=0.1, precision=2 ** 0)
    print(g.leakage)
